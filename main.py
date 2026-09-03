import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

# 1. Initialize the core stable MCP Server
mcp_server = Server("torch-mobile-mcp")

# 2. Re-register the tools using the direct function syntax to avoid registration crashes
@mcp_server.list_tools()
async def handle_list_tools():
    return [
        Tool(
            name="check_torch_env",
            description="Checks the remote server's PyTorch version and setup environment.",
            inputSchema={"type": "object", "properties": {}}
        )
    ]

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

# 3. Setup the FastAPI app framework
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Bind the SSE transport routing pathways properly
sse_transport = SseServerTransport("/messages")

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Use /sse endpoint"}

@app.get("/sse")
@app.get("/sse/")
async def handle_sse(request: Request):
    async with sse_transport.connect_sse(request.scope, request.receive, request.send) as queue:
        await mcp_server.run(queue, sse_transport.extra_context())

@app.post("/messages")
@app.post("/messages/")
async def handle_messages(request: Request):
    await sse_transport.handle_post_message(request)
