import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.server.fastapi import FastHtmlServer
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
# This assigns the background event queue directly to ensure it hooks into FastAPI
@app.on_event("startup")
async def startup_event():
    app.state.mcp_server = mcp_server

# 6. Bind the SSE transport pathways manually to avoid import mapping errors
from mcp.server.sse import SseServerTransport
sse_transport = SseServerTransport("/messages")

@app.get("/sse")
@app.get("/sse/")
async def handle_sse(request: Request):
    async with sse_transport.connect_sse(request.scope, request.receive, request.send) as queue:
        await mcp_server.run(queue, sse_transport.extra_context())

@app.post("/messages")
@app.post("/messages/")
async def handle_messages(request: Request):
    await sse_transport.handle_post_message(request)
