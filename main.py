import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

# Initialize the stable core MCP Server
mcp_server = Server("torch-mobile-mcp")

# Define the tools array manually for the version 1 specification
AVAILABLE_TOOLS = [
    Tool(
        name="check_torch_env",
        description="Checks the remote server's PyTorch version and setup environment.",
        inputSchema={"type": "object", "properties": {}}
    )
]

@mcp_server.list_tools()
async def handle_list_tools():
    return AVAILABLE_TOOLS

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "check_torch_env":
        return [
            TextContent(
                type="text",
                text=f"PyTorch version {torch.__version__} is running successfully on CPU."
            )
        ]
    raise ValueError(f"Unknown tool: {name}")

# --- SSE TRANSPORT INTEGRATION ---
sse_transport = SseServerTransport("/messages")
app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url="/sse")

@app.get("/sse")
async def handle_sse(request: Request):
    async with sse_transport.connect_sse(request.scope, request.receive, request.send) as queue:
        await mcp_server.run(queue, sse_transport.extra_context())

@app.post("/messages")
async def handle_messages(request: Request):
    await sse_transport.handle_post_message(request)
