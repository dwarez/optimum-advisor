use std::{ffi::OsString, time::Duration};

use crate::{
    config::RuntimeConfig,
    domain::run::{GpuRecord, HardwareProfile},
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{
        cancel::CancellationToken,
        process::{ProcessCapture, ProcessExecutor, ProcessFailure, ProcessSpec},
    },
};

const HARDWARE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_HARDWARE_OUTPUT_BYTES: u64 = 256 * 1024;

pub(crate) fn inspect_hardware(
    runtime: &RuntimeConfig,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<HardwareProfile> {
    let (all_gpus, warnings) = match query_nvidia_smi(true, executor, cancellation)
        .and_then(|text| parse_nvidia_smi_csv(&text, true))
    {
        Ok(gpus) => (gpus, Vec::new()),
        Err(compute_error) => {
            let gpus = query_nvidia_smi(false, executor, cancellation)
                .and_then(|text| parse_nvidia_smi_csv(&text, false))?;
            (
                gpus,
                vec![format!("compute capability unavailable: {compute_error}")],
            )
        }
    };
    if all_gpus.is_empty() {
        return Err(hardware_error("nvidia-smi reported no GPUs"));
    }

    let selected_gpus = if runtime.gpu_devices.is_empty() {
        if all_gpus.len() < runtime.gpus {
            return Err(hardware_error(format!(
                "requested {} GPUs but nvidia-smi reported {}",
                runtime.gpus,
                all_gpus.len()
            )));
        }
        all_gpus.iter().take(runtime.gpus).cloned().collect()
    } else {
        runtime
            .gpu_devices
            .iter()
            .map(|requested| {
                all_gpus
                    .iter()
                    .find(|gpu| gpu.index.to_string() == *requested || gpu.uuid == *requested)
                    .cloned()
                    .ok_or_else(|| {
                        hardware_error(format!(
                            "configured GPU device {requested:?} was not reported by nvidia-smi"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?
    };

    Ok(HardwareProfile {
        source: "nvidia-smi".to_string(),
        cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .filter(|value| !value.trim().is_empty()),
        all_gpus,
        selected_gpus,
        warnings,
    })
}

pub(crate) fn format_hardware_profile(profile: &HardwareProfile) -> String {
    let mut text = format!("source: {}\n", profile.source);
    if let Some(value) = &profile.cuda_visible_devices {
        text.push_str(&format!("cuda_visible_devices: {value}\n"));
    }
    text.push_str(&format!("gpus: {}\n", profile.all_gpus.len()));
    for gpu in &profile.all_gpus {
        let selected = profile
            .selected_gpus
            .iter()
            .any(|selected| selected.uuid == gpu.uuid);
        text.push_str(&format!(
            "gpu[{}]: selected={} name={:?} uuid={} compute_capability={} memory_total_mib={} memory_free_mib={} memory_used_mib={}\n",
            gpu.index,
            selected,
            gpu.name,
            gpu.uuid,
            gpu.compute_capability.as_deref().unwrap_or("unknown"),
            gpu.memory_total_mib,
            gpu.memory_free_mib,
            gpu.memory_used_mib,
        ));
    }
    for warning in &profile.warnings {
        text.push_str(&format!("warning: {warning}\n"));
    }
    text
}

fn query_nvidia_smi(
    include_compute_capability: bool,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<String> {
    let fields = if include_compute_capability {
        "index,name,uuid,compute_cap,memory.total,memory.free,memory.used"
    } else {
        "index,name,uuid,memory.total,memory.free,memory.used"
    };
    let mut spec = ProcessSpec::new(
        "nvidia-smi",
        [
            OsString::from(format!("--query-gpu={fields}")),
            OsString::from("--format=csv,noheader,nounits"),
        ],
    )
    .with_stage(ExecutionStage::Preflight)
    .with_timeout(HARDWARE_TIMEOUT)
    .with_safe_display("nvidia-smi <hardware query>");
    spec.max_stdout_bytes = MAX_HARDWARE_OUTPUT_BYTES;
    spec.max_stderr_bytes = MAX_HARDWARE_OUTPUT_BYTES;
    let outcome = executor
        .execute(&spec, cancellation)
        .map_err(map_hardware_failure)?;
    let ProcessCapture::Artifacts(capture) = outcome.capture else {
        return Err(hardware_error(
            "hardware inspection unexpectedly used secret capture",
        ));
    };
    if capture.stdout.truncated {
        return Err(Error::new(
            ErrorKind::OutputTruncated,
            Some(ExecutionStage::Preflight),
            "nvidia-smi output exceeded 256 KiB",
        ));
    }
    Ok(capture.stdout.tail)
}

fn parse_nvidia_smi_csv(text: &str, has_compute_capability: bool) -> Result<Vec<GpuRecord>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse_gpu_line(line, has_compute_capability))
        .collect()
}

fn parse_gpu_line(line: &str, has_compute_capability: bool) -> Result<GpuRecord> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    let expected = if has_compute_capability { 7 } else { 6 };
    if fields.len() != expected {
        return Err(hardware_error(format!(
            "unexpected nvidia-smi row shape: expected {expected} columns, got {}",
            fields.len()
        )));
    }
    let memory_offset = if has_compute_capability { 4 } else { 3 };
    Ok(GpuRecord {
        index: parse_number(fields[0], "GPU index")?,
        name: required(fields[1], "GPU name")?.to_string(),
        uuid: required(fields[2], "GPU UUID")?.to_string(),
        compute_capability: has_compute_capability
            .then(|| optional(fields[3]))
            .flatten(),
        memory_total_mib: parse_number(fields[memory_offset], "memory.total")?,
        memory_free_mib: parse_number(fields[memory_offset + 1], "memory.free")?,
        memory_used_mib: parse_number(fields[memory_offset + 2], "memory.used")?,
    })
}

fn required<'a>(value: &'a str, field: &str) -> Result<&'a str> {
    if value.is_empty() || matches!(value, "N/A" | "[Not Supported]") {
        Err(hardware_error(format!("nvidia-smi returned no {field}")))
    } else {
        Ok(value)
    }
}

fn optional(value: &str) -> Option<String> {
    (!value.is_empty() && !matches!(value, "N/A" | "[Not Supported]")).then(|| value.to_string())
}

fn parse_number<T: std::str::FromStr>(value: &str, field: &str) -> Result<T> {
    value
        .split_whitespace()
        .next()
        .unwrap_or(value)
        .parse()
        .map_err(|_| hardware_error(format!("nvidia-smi {field} has invalid value: {value}")))
}

fn map_hardware_failure(failure: ProcessFailure) -> Error {
    hardware_error("nvidia-smi hardware query failed")
        .with_operation("inspect NVIDIA hardware")
        .with_source(failure.error)
}

fn hardware_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Validation,
        Some(ExecutionStage::Preflight),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete_nvidia_smi_rows_strictly() {
        let gpus = parse_nvidia_smi_csv(
            "0, NVIDIA L4, GPU-abc, 8.9, 23034, 21000, 2034\n1, NVIDIA L4, GPU-def, 8.9, 23034, 23000, 34\n",
            true,
        )
        .unwrap();

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].compute_capability.as_deref(), Some("8.9"));
        assert_eq!(gpus[0].memory_total_mib, 23034);
        assert!(parse_nvidia_smi_csv("0, missing columns", true).is_err());
    }
}
