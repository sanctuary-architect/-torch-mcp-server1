import os
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.types import Tool, TextContent

# 1. Initialize the core stable MCP Server
mcp_server = Server("torch-mobile-mcp")

# 2. Register tools exactly how the SDK expects them
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

# 3. Create a standard FastAPI application instance
app = FastAPI()

# 4. Apply global CORS settings so Claude's mobile infrastructure is authorized
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Initialize the server transport pathway cleanly
@app.on_event("startup")
async def startup_event():
    app.state.mcp_server = mcp_server

# 6. Bind the SSE transport pathways manually with explicit Request references
from mcp.server.sse import SseServerTransport
sse_transport = SseServerTransport("/messages")

@app.get("/sse")
@app.get("/sse/")
async def handle_sse(request: Request):
    # connect_sse is an async context manager that yields a (read_stream,
    # write_stream) TUPLE — not a single object — and Server.run() needs
    # both streams plus a set of initialization options as its third
    # argument. Neither of those was true in the previous version.
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )

@app.post("/messages")
@app.post("/messages/")
async def handle_messages(request: Request):
    # handle_post_message is itself a raw ASGI app — it expects
    # (scope, receive, send), the same way connect_sse does. Passing it a
    # FastAPI Request object directly hits the exact same class of error
    # ('Request' object has no attribute X) that broke /sse.
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
