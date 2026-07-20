#!/usr/bin/env python3
"""Dummy MCP client for probing the optimum-advisor MCP server.

Speaks newline-delimited JSON-RPC 2.0 over stdio, exactly like an MCP agent
host. Stdlib only.

Usage:
  scripts/mcp-probe.py [--binary PATH] info
  scripts/mcp-probe.py [--binary PATH] list
  scripts/mcp-probe.py [--binary PATH] call TOOL [JSON_ARGS]

Examples:
  scripts/mcp-probe.py list
  scripts/mcp-probe.py call rank_candidates \
    '{"metric":"tps","candidates":[{"id":"a","value":1.0}]}'
"""

import argparse
import itertools
import json
import queue
import subprocess
import sys
import threading


class McpClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.queue = queue.Queue()
        self.ids = itertools.count(1)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.queue.put(json.loads(line))

    def notify(self, method, params=None):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=3600):
        request_id = next(self.ids)
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        while True:
            response = self.queue.get(timeout=timeout)
            if response.get("id") == request_id:
                return response

    def handshake(self):
        result = self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "mcp-probe", "version": "0"},
            },
        )
        self.notify("notifications/initialized")
        return result

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/debug/optimum-advisor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("info")
    subcommands.add_parser("list")
    call = subcommands.add_parser("call")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?", default="{}")
    args = parser.parse_args()

    client = McpClient([args.binary, "mcp"])
    try:
        init = client.handshake()
        if args.command == "info":
            print(json.dumps(init["result"], indent=2))
        elif args.command == "list":
            tools = client.request("tools/list")["result"]["tools"]
            for tool in tools:
                annotations = tool.get("annotations", {})
                flags = ",".join(
                    key.replace("Hint", "")
                    for key, value in annotations.items()
                    if value is True
                )
                print(f"{tool['name']} [{flags}]\n  {tool['description']}\n")
        elif args.command == "call":
            response = client.request(
                "tools/call",
                {"name": args.tool, "arguments": json.loads(args.arguments)},
            )
            print(json.dumps(response.get("result", response.get("error")), indent=2))
            if response.get("result", {}).get("isError"):
                sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
