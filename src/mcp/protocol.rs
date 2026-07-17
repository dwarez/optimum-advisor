use std::{
    io::{BufRead, Write},
    sync::{
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
        Arc, Mutex, MutexGuard,
    },
    thread,
    time::Duration,
};

use serde_json::{json, Value};

use crate::{
    error::{Error, ErrorKind, Result},
    runtime::cancel::CancellationToken,
};

use super::{schema::tool_definitions, tools::call_tool_request};

const PROTOCOL_VERSION: &str = "2025-11-25";
/// Older protocol revisions this server is wire-compatible with: everything
/// it uses beyond them (tool annotations, `structuredContent`,
/// `outputSchema`) is additive, so older clients simply ignore those fields.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26"];
pub(super) const SERVER_NAME: &str = "optimum-advisor";
pub(super) const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LifecycleState {
    #[default]
    AwaitingInitialize,
    AwaitingInitializedNotification,
    Ready,
}

#[derive(Default)]
struct Lifecycle {
    state: LifecycleState,
    disconnect: bool,
}

#[derive(Default)]
pub(super) struct InFlightState {
    pub(super) active: Option<(Value, CancellationToken)>,
    pending_cancellations: Vec<Value>,
}

enum ReaderEvent {
    Message(Value),
    Response(Value),
    Failure(Error),
    Eof,
}

pub(crate) fn serve(input: impl BufRead + Send + 'static, mut output: impl Write) -> Result<()> {
    let os_cancellation = CancellationToken::new();
    os_cancellation.register_os_signals()?;
    let in_flight = Arc::new(Mutex::new(InFlightState::default()));
    let (sender, receiver) = mpsc::channel();
    let reader_in_flight = Arc::clone(&in_flight);
    thread::Builder::new()
        .name("optimum-advisor-mcp-reader".into())
        .spawn(move || read_messages(input, sender, reader_in_flight))
        .map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to start MCP reader thread").with_source(source)
        })?;

    dispatch_messages(receiver, &mut output, &os_cancellation, &in_flight)
}

fn dispatch_messages(
    receiver: Receiver<ReaderEvent>,
    output: &mut impl Write,
    os_cancellation: &CancellationToken,
    in_flight: &Arc<Mutex<InFlightState>>,
) -> Result<()> {
    let mut lifecycle = Lifecycle::default();
    loop {
        if os_cancellation.is_cancelled() {
            return Err(Error::new(
                ErrorKind::Interrupted,
                None,
                "MCP server interrupted",
            ));
        }
        let event = match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        };
        match event {
            ReaderEvent::Eof => return Ok(()),
            ReaderEvent::Failure(error) => return Err(error),
            ReaderEvent::Response(response) => write_message(output, &response)?,
            ReaderEvent::Message(request) => {
                if let Some(response) =
                    handle_request(&request, os_cancellation, &mut lifecycle, in_flight)
                {
                    if os_cancellation.is_cancelled() {
                        return Err(Error::new(
                            ErrorKind::Interrupted,
                            None,
                            "MCP server interrupted",
                        ));
                    }
                    write_message(output, &response)?;
                }
                if lifecycle.disconnect {
                    return Ok(());
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LineRead {
    Eof,
    Line,
    TooLong,
}

fn read_bounded_line(input: &mut impl BufRead, line: &mut Vec<u8>) -> Result<LineRead> {
    line.clear();
    let mut too_long = false;
    loop {
        let available = input.fill_buf().map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to read MCP request stream").with_source(source)
        })?;
        if available.is_empty() {
            return if too_long {
                Ok(LineRead::TooLong)
            } else if line.is_empty() {
                Ok(LineRead::Eof)
            } else {
                Ok(LineRead::Line)
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        if !too_long {
            if line.len().saturating_add(consumed) > MAX_REQUEST_BYTES {
                line.clear();
                too_long = true;
            } else {
                line.extend_from_slice(&available[..consumed]);
            }
        }
        input.consume(consumed);
        if newline.is_some() {
            if too_long {
                return Ok(LineRead::TooLong);
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(LineRead::Line);
        }
    }
}

fn read_messages(
    mut input: impl BufRead,
    sender: Sender<ReaderEvent>,
    in_flight: Arc<Mutex<InFlightState>>,
) {
    let mut line = Vec::new();
    loop {
        let read = match read_bounded_line(&mut input, &mut line) {
            Ok(read) => read,
            Err(error) => {
                let _ = sender.send(ReaderEvent::Failure(error));
                return;
            }
        };
        let event = match read {
            LineRead::Eof => {
                let _ = sender.send(ReaderEvent::Eof);
                return;
            }
            LineRead::TooLong => ReaderEvent::Response(rpc_error(
                Value::Null,
                -32700,
                format!("request exceeds {MAX_REQUEST_BYTES} bytes"),
            )),
            LineRead::Line if line.iter().all(u8::is_ascii_whitespace) => continue,
            LineRead::Line => match serde_json::from_slice::<Value>(&line) {
                Ok(request) => {
                    if let Some(request_id) = cancelled_request_id(&request) {
                        cancel_request(&in_flight, request_id);
                        continue;
                    }
                    ReaderEvent::Message(request)
                }
                Err(error) => ReaderEvent::Response(rpc_error(
                    Value::Null,
                    -32700,
                    format!("parse error: {error}"),
                )),
            },
        };
        if sender.send(event).is_err() {
            return;
        }
    }
}

fn cancelled_request_id(request: &Value) -> Option<Value> {
    let request = request.as_object()?;
    if request.contains_key("id")
        || request.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
        || request.get("method").and_then(Value::as_str) != Some("notifications/cancelled")
    {
        return None;
    }
    request
        .get("params")
        .and_then(Value::as_object)?
        .get("requestId")
        .filter(|id| valid_request_id(id))
        .cloned()
}

pub(super) fn cancel_request(in_flight: &Arc<Mutex<InFlightState>>, request_id: Value) {
    const MAX_PENDING_CANCELLATIONS: usize = 64;
    let mut state = lock_in_flight(in_flight);
    if let Some((active_id, cancellation)) = state.active.as_ref() {
        if active_id == &request_id {
            cancellation.cancel();
            return;
        }
    }
    if state.pending_cancellations.len() < MAX_PENDING_CANCELLATIONS
        && !state.pending_cancellations.contains(&request_id)
    {
        state.pending_cancellations.push(request_id);
    }
}

pub(super) fn begin_request(
    in_flight: &Arc<Mutex<InFlightState>>,
    request_id: &Value,
    parent: &CancellationToken,
) -> CancellationToken {
    let cancellation = parent.child();
    let mut state = lock_in_flight(in_flight);
    if let Some(index) = state
        .pending_cancellations
        .iter()
        .position(|pending| pending == request_id)
    {
        state.pending_cancellations.swap_remove(index);
        cancellation.cancel();
    }
    state.active = Some((request_id.clone(), cancellation.clone()));
    cancellation
}

pub(super) fn finish_request(in_flight: &Arc<Mutex<InFlightState>>, request_id: &Value) {
    let mut state = lock_in_flight(in_flight);
    if state
        .active
        .as_ref()
        .is_some_and(|(active_id, _)| active_id == request_id)
    {
        state.active = None;
    }
}

pub(super) fn lock_in_flight(
    in_flight: &Arc<Mutex<InFlightState>>,
) -> MutexGuard<'_, InFlightState> {
    in_flight
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn handle_request(
    request: &Value,
    os_cancellation: &CancellationToken,
    lifecycle: &mut Lifecycle,
    in_flight: &Arc<Mutex<InFlightState>>,
) -> Option<Value> {
    let Some(request) = request.as_object() else {
        return Some(rpc_error(Value::Null, -32600, "request must be an object"));
    };
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return request.contains_key("id").then(|| {
            rpc_error(
                request["id"].clone(),
                -32600,
                "request method must be a string",
            )
        });
    };
    let Some(id) = request.get("id").cloned() else {
        if request.get("jsonrpc").and_then(Value::as_str) == Some("2.0")
            && method == "notifications/initialized"
            && lifecycle.state == LifecycleState::AwaitingInitializedNotification
        {
            lifecycle.state = LifecycleState::Ready;
        }
        return None;
    };
    if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(rpc_error(id, -32600, "jsonrpc must be \"2.0\""));
    }
    if !valid_request_id(&id) {
        return Some(rpc_error(Value::Null, -32600, "invalid request id"));
    }
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
    if method == "ping" {
        return Some(json!({ "jsonrpc": "2.0", "id": id, "result": {} }));
    }
    if method == "initialize" {
        if lifecycle.state != LifecycleState::AwaitingInitialize {
            return Some(rpc_error(id, -32600, "initialize may be called only once"));
        }
        let params = match initialize_params(&params) {
            Ok(params) => params,
            Err(message) => {
                lifecycle.disconnect = true;
                return Some(rpc_error(id, -32602, message));
            }
        };
        // Spec-mandated negotiation: echo the requested version when this
        // server supports it; otherwise answer with the latest version it
        // speaks and let the client decide whether to continue. A mismatch is
        // never a server-side error.
        let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&params) {
            params
        } else {
            PROTOCOL_VERSION
        };
        lifecycle.state = LifecycleState::AwaitingInitializedNotification;
        return Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": initialize(negotiated)
        }));
    }
    if lifecycle.state != LifecycleState::Ready {
        return Some(rpc_error(
            id,
            -32002,
            "server is not ready; initialize and send notifications/initialized",
        ));
    }
    let result = match method {
        "tools/list" => tool_definitions().map(|tools| json!({ "tools": tools })),
        "tools/call" => {
            let cancellation = begin_request(in_flight, &id, os_cancellation);
            let result = call_tool_request(&params, &cancellation);
            finish_request(in_flight, &id);
            result
        }
        _ => return Some(rpc_error(id, -32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(error) => rpc_error(id, -32602, error.to_string()),
    })
}

fn valid_request_id(id: &Value) -> bool {
    id.is_null() || id.is_string() || id.is_number()
}

fn initialize_params(params: &Value) -> std::result::Result<&str, String> {
    let params = params
        .as_object()
        .ok_or_else(|| "initialize params must be an object".to_string())?;
    if let Some(unknown) = params.keys().find(|key| {
        !matches!(
            key.as_str(),
            "protocolVersion" | "capabilities" | "clientInfo"
        )
    }) {
        return Err(format!("unknown initialize parameter: {unknown}"));
    }
    match params.get("protocolVersion") {
        Some(Value::String(version)) => Ok(version),
        Some(_) => Err("initialize protocolVersion must be a string".into()),
        None => Ok(PROTOCOL_VERSION),
    }
}

fn initialize(negotiated_version: &str) -> Value {
    json!({
        "protocolVersion": negotiated_version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Bounded inspection and evaluation tools for LLM serving configurations"
        },
        "instructions": "Find the best serving configuration for a model by measuring candidates. Typical workflow: inspect_hardware to learn the GPUs; inspect_engine to learn which serving arguments the image accepts; validate_config to check a candidate cheaply; then run_sweep (several candidates via [sweep] arrays) or evaluate_candidate (one candidate), which serve, optionally quality-check, and benchmark each candidate and return per-trial summaries; get_report and list_runs read results back later; rank_candidates orders observations you already hold. Execution tools run for many minutes and requests are processed sequentially. Configs use the same shape as schema-v2 TOML files. Tool failures are typed payloads with kind/stage/message."
    })
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() }
    })
}

fn write_message(output: &mut impl Write, message: &Value) -> Result<()> {
    let bytes = serde_json::to_vec(message).map_err(|source| {
        Error::new(ErrorKind::Protocol, None, "failed to encode MCP response").with_source(source)
    })?;
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(Error::new(
            ErrorKind::OutputTruncated,
            None,
            format!("MCP response exceeds {MAX_RESPONSE_BYTES} bytes"),
        ));
    }
    output.write_all(&bytes).map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to write MCP response").with_source(source)
    })?;
    output.write_all(b"\n").map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to terminate MCP response").with_source(source)
    })?;
    output.flush().map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to flush MCP response").with_source(source)
    })
}
