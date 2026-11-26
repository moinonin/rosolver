"""
Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio.

Supports both standard MCP newline-delimited UTF-8 JSON-RPC 2.0 transport (`2024-11-05` /
`2025-03-26`) and `Content-Length` header framing automatically, redirecting any stray
library stdout during tool execution to `stderr` so protocol frames remain 100% pure JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from typing import Any, BinaryIO, Dict, Mapping, Optional, Sequence, Tuple

import rosolver
from ..results import to_jsonable
from .tools import (
    TOOL_HANDLERS,
    TOOL_SPECIFICATIONS,
    get_mcp_prompt,
    list_mcp_prompts,
    list_mcp_resources,
    read_mcp_resource,
)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

SERVER_INSTRUCTIONS = (
    "Rosolver MCP evaluates scientific equations, dimensions, calibrations, ODE/PDE/Stefan "
    "simulations, MMS/GCI/conservation verifications, experimental validations, and reports. "
    "AI proposes; Rosolver deterministically evaluates. Always surface Failed, Inconclusive, "
    "or Warning statuses verbatim and never claim formal regulatory certification."
)


class MCPServer:
    """Deterministic JSON-RPC 2.0 MCP server dispatching Rosolver tools, resources, and prompts."""

    def __init__(self) -> None:
        self.initialized = False

    def handle_message(self, message: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(message, Mapping) or message.get("jsonrpc") != "2.0":
            return self._error(message.get("id") if isinstance(message, Mapping) else None, -32600, "Invalid JSON-RPC 2.0 request.")

        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications have no 'id' and must not receive a response
        if "id" not in message:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        if not isinstance(method, str):
            return self._error(msg_id, -32600, "Missing or non-string 'method'.")

        try:
            if method == "initialize":
                req_ver = str(params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION))
                neg_ver = req_ver if req_ver in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
                return self._result(
                    msg_id,
                    {
                        "protocolVersion": neg_ver,
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                            "prompts": {"listChanged": False},
                        },
                        "serverInfo": {
                            "name": "rosolver-mcp",
                            "version": rosolver.__version__,
                        },
                        "instructions": SERVER_INSTRUCTIONS,
                    },
                )

            if method == "ping":
                return self._result(msg_id, {})

            if method == "tools/list":
                return self._result(msg_id, {"tools": TOOL_SPECIFICATIONS})

            if method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments") or {}
                if not isinstance(tool_name, str) or tool_name not in TOOL_HANDLERS:
                    return self._error(msg_id, -32602, f"Unknown tool '{tool_name}'.")
                if not isinstance(tool_args, Mapping):
                    return self._error(msg_id, -32602, "Tool 'arguments' must be an object.")
                return self._call_tool(msg_id, tool_name, tool_args)

            if method == "resources/list":
                return self._result(msg_id, {"resources": list_mcp_resources()})

            if method == "resources/read":
                uri = params.get("uri")
                if not isinstance(uri, str):
                    return self._error(msg_id, -32602, "Missing string 'uri' parameter.")
                try:
                    mime, content = read_mcp_resource(uri)
                except KeyError as exc:
                    return self._error(msg_id, -32602, str(exc))
                return self._result(
                    msg_id,
                    {"contents": [{"uri": uri, "mimeType": mime, "text": content}]},
                )

            if method == "prompts/list":
                return self._result(msg_id, {"prompts": list_mcp_prompts()})

            if method == "prompts/get":
                pname = params.get("name")
                if not isinstance(pname, str):
                    return self._error(msg_id, -32602, "Missing string 'name' parameter.")
                try:
                    prompt_obj = get_mcp_prompt(pname, params.get("arguments"))
                except KeyError as exc:
                    return self._error(msg_id, -32602, str(exc))
                return self._result(msg_id, prompt_obj)

            return self._error(msg_id, -32601, f"Method not found: '{method}'.")

        except Exception as exc:  # pragma: no cover - defensive protocol boundary
            return self._error(msg_id, -32603, f"Internal server error: {type(exc).__name__}: {exc}")

    def _call_tool(
        self,
        msg_id: Any,
        tool_name: str,
        tool_args: Mapping[str, Any],
    ) -> Dict[str, Any]:
        handler = TOOL_HANDLERS[tool_name]
        stray_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(stray_out):
                payload = to_jsonable(handler(tool_args))
            extra = stray_out.getvalue().strip()
            if extra:
                print(extra, file=sys.stderr)
            text = json.dumps(payload, indent=2)
            return self._result(
                msg_id,
                {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": payload,
                    "isError": False,
                },
            )
        except Exception as exc:
            err_payload = {
                "error": type(exc).__name__,
                "message": str(exc),
                "tool": tool_name,
            }
            return self._result(
                msg_id,
                {
                    "content": [{"type": "text", "text": json.dumps(err_payload, indent=2)}],
                    "structuredContent": err_payload,
                    "isError": True,
                },
            )

    @staticmethod
    def _result(msg_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _read_message(stream: BinaryIO) -> Tuple[Optional[bytes], bool]:
    """
    Read one JSON-RPC message from binary `stream`.
    Returns `(raw_bytes, used_content_length_framing)` or `(None, False)` on EOF.
    """
    while True:
        first_line = stream.readline()
        if not first_line:
            return None, False
        stripped = first_line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length:"):
            length = int(stripped.split(b":", 1)[1].strip())
            # Consume remaining headers until blank line
            while True:
                hdr = stream.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
            body = stream.read(length)
            return body, True
        return stripped, False


def _write_message(stream: BinaryIO, payload: Dict[str, Any], framed: bool) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if framed:
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        stream.write(header + body)
    else:
        stream.write(body + b"\n")
    stream.flush()


def serve_stdio(stdin: Optional[BinaryIO] = None, stdout: Optional[BinaryIO] = None) -> int:
    """Run the MCP JSON-RPC 2.0 server loop over binary stdio streams."""
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer
    server = MCPServer()

    while True:
        raw, framed = _read_message(in_stream)
        if raw is None:
            return 0
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            err = server._error(None, -32700, f"Parse error: {exc}")
            _write_message(out_stream, err, framed)
            continue

        if isinstance(decoded, list):
            responses = [r for item in decoded if (r := server.handle_message(item)) is not None]
            if responses:
                body = json.dumps(responses, separators=(",", ":")).encode("utf-8")
                if framed:
                    out_stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
                else:
                    out_stream.write(body + b"\n")
                out_stream.flush()
        else:
            resp = server.handle_message(decoded)
            if resp is not None:
                _write_message(out_stream, resp, framed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosolver-mcp",
        description="Run the Rosolver Model Context Protocol (MCP) server over stdio.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rosolver-mcp {rosolver.__version__}",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print registered MCP tool schemas as JSON and exit.",
    )
    args = parser.parse_args(argv)
    if args.list_tools:
        print(json.dumps({"tools": TOOL_SPECIFICATIONS}, indent=2))
        return 0
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
