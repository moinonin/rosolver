"""Model Context Protocol (MCP) server and AI-native scientific tool surface for Rosolver."""

from .server import MCPServer, main, serve_stdio
from .tools import (
    MCPToolError,
    TOOL_HANDLERS,
    TOOL_SPECIFICATIONS,
    get_mcp_prompt,
    list_mcp_prompts,
    list_mcp_resources,
    read_mcp_resource,
)

__all__ = [
    "MCPServer",
    "MCPToolError",
    "TOOL_HANDLERS",
    "TOOL_SPECIFICATIONS",
    "get_mcp_prompt",
    "list_mcp_prompts",
    "list_mcp_resources",
    "main",
    "read_mcp_resource",
    "serve_stdio",
]
