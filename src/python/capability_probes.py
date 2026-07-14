import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base_url, model, artifact_path, tool_parser, reasoning_parser, timeout = sys.argv[1:]
timeout = max(1, int(timeout))
checks = []


def post(payload):
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read(1048577)
            if len(data) > 1048576:
                raise RuntimeError("capability response exceeded 1 MiB")
            return json.loads(data)
    except HTTPError as error:
        data = error.read(4096).decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {data}") from error


def run(domain, parser, probe):
    try:
        probe()
        checks.append({"domain": domain, "parser": parser, "passed": True})
    except Exception:
        checks.append({"domain": domain, "parser": parser, "passed": False})


def tool_calling():
    response = post(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Call get_temperature once for Rome."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_temperature",
                        "description": "Get a city temperature.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        }
    )
    calls = response["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1
    function = calls[0]["function"]
    assert function["name"] == "get_temperature"
    arguments = function["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    assert isinstance(arguments.get("city"), str)


def reasoning():
    response = post(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "What is 17 multiplied by 19?"}
            ],
            "temperature": 0,
            "max_tokens": 256,
        }
    )
    message = response["choices"][0]["message"]
    reasoning_content = message.get("reasoning") or message.get("reasoning_content")
    assert isinstance(reasoning_content, str) and reasoning_content.strip()
    content = message.get("content") or ""
    assert "<think>" not in content and "</think>" not in content


if tool_parser:
    run("tool_calling", tool_parser, tool_calling)
if reasoning_parser:
    run("reasoning", reasoning_parser, reasoning)

path = Path(artifact_path)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("x") as output:
    json.dump({"schema_version": 1, "checks": checks}, output, indent=2)
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
