import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server import Server
from mcp.server.fastapi import FastApiServer
from mcp.types import Tool, TextContent

# 1. Create the core MCP Server instance
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

# 5. Initialize the official FastApiServer wrapper to automatically manage the SSE handshake
# This creates /sse and /messages endpoints natively with built-in initialization handling
mcp_fastapi_wrapper = FastApiServer(mcp_server)

# Link the protocol wrapper directly into our web application router paths
app.include_router(mcp_fastapi_wrapper.router, prefix="")
