import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Route, Mount
from starlette.responses import Response
from mcp.server import Server
from mcp.server.sse import SseServerTransport
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

# 3. SSE transport. Note the trailing slash on the messages endpoint — both
#    the transport's own endpoint string and the Mount path below need to
#    match, since Starlette's Mount is sensitive to trailing-slash mismatches.
sse_transport = SseServerTransport("/messages/")


async def handle_sse(request):
    """
    A plain Request -> Response function, per Starlette's Route contract —
    but the actual response is streamed and fully completed *inside*
    connect_sse via the raw ASGI send() it's handed directly
    (request._send). The `return Response()` at the end exists only to
    satisfy Route's expectation that *something* comes back once the SSE
    connection has ended; per the MCP SDK's own docs, omitting it raises
    ``TypeError: 'NoneType' object is not callable`` on client disconnect.
    """
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )
    return Response()


# 4. FastAPI app — used for CORS and would be used for any ordinary
#    request/response routes, but deliberately NOT for /sse or /messages.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Mount the MCP transport as raw Starlette Route/Mount objects, appended
#    directly onto FastAPI's underlying router. This is deliberate, not an
#    oversight: FastAPI's @app.get/@app.post decorators route every return
#    value (including None) through FastAPI's own response-serialization
#    layer, which tries to send a *second* response after connect_sse and
#    handle_post_message have already completed one via their own raw
#    ASGI send() calls — producing "Unexpected ASGI message
#    'http.response.start' sent, after response already completed."
#    Plain Starlette Route/Mount objects bypass that layer entirely, which
#    is why every official MCP SSE example uses Starlette directly for
#    these two endpoints rather than a framework's higher-level decorators.
app.router.routes.append(Route("/sse", endpoint=handle_sse, methods=["GET"]))
app.router.routes.append(Mount("/messages/", app=sse_transport.handle_post_message))
