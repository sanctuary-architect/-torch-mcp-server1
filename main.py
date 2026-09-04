import os
import copy
import traceback
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


# ---------------------------------------------------------------------------
# The GuardrailCallback class below is copied verbatim, byte-for-byte, from
# the TensorScript transpiler's actual output for `optimize BaseLM as
# SFTStage` (see tensorscript_v1_spec.md §7). It is NOT reimplemented or
# paraphrased here — that matters, because the whole point of this tool is
# to verify the real generated artifact, not a hand-written approximation
# of it. The only change from the transpiled version is the base class:
# the real generated code subclasses transformers.TrainerCallback, but
# this server only has torch installed (confirmed via check_torch_env),
# not the full transformers/peft/datasets stack. _MinimalCallbackBase is a
# zero-behavior stand-in with the same shape, used the same way the
# 'stubs/transformers.py' TrainerCallback stub was used in the original
# numpy-based verification — except everything downstream of it here is
# real torch, not a simulation of torch.
class _MinimalCallbackBase:
    pass


class GuardrailCallback(_MinimalCallbackBase):
    """Auto-generated from the TensorScript guardrails: list.
    Evaluates all rules each step, in declaration order, against the
    latest logged metrics. terminate_early short-circuits remaining
    rules for that step. Multiple LR-scaling triggers in one step
    compose multiplicatively rather than overwriting each other.
    rollback_and_scale_lr performs a genuine in-place weight restore
    from the most recent on_save snapshot, once one exists."""

    RULES = [   {   'action': {'args': [{'key': None, 'value': 0.5}], 'call': 'rollback_and_scale_lr'},
            'block': None,
            'duration_steps': None,
            'epsilon': None,
            'metric': 'loss',
            'op': '>',
            'value': 2.5},
        {   'action': 'terminate_early',
            'block': None,
            'duration_steps': 200,
            'epsilon': 0.0001,
            'metric': 'loss',
            'op': 'flat_lines',
            'value': None}]

    def __init__(self):
        self._metric_history = {}
        self._consecutive_true = {}
        self._has_checkpoint = False
        self._last_checkpoint_state = None

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            self._last_checkpoint_state = copy.deepcopy(model.state_dict())
            self._has_checkpoint = True
        return control

    def _flat_lines(self, metric, epsilon, window):
        hist = self._metric_history.get(metric, [])
        if len(hist) < window:
            return False
        recent = hist[-window:]
        variance = sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)
        return variance < epsilon

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self._metric_history.setdefault(k, []).append(v)

        lr_scale_factor = 1.0
        for rule_idx, rule in enumerate(self.RULES):
            metric = rule["metric"]
            if metric not in logs:
                continue
            triggered = False
            if rule["op"] == "flat_lines":
                window = rule["duration_steps"] or 1
                triggered = self._flat_lines(metric, rule["epsilon"], window)
            else:
                val = logs[metric]
                op = rule["op"]
                threshold = rule["value"]
                triggered = {
                    ">": val > threshold, "<": val < threshold,
                    ">=": val >= threshold, "<=": val <= threshold,
                    "==": val == threshold,
                }.get(op, False)
                if triggered and rule["duration_steps"]:
                    key = (rule_idx, "streak")
                    self._consecutive_true[key] = self._consecutive_true.get(key, 0) + 1
                    triggered = self._consecutive_true[key] >= rule["duration_steps"]
                elif rule["duration_steps"]:
                    self._consecutive_true[(rule_idx, "streak")] = 0

            if not triggered:
                continue

            action = rule["action"]
            action_name = action["call"] if isinstance(action, dict) else action

            if action_name == "terminate_early":
                control.should_training_stop = True
                break

            elif action_name == "rollback_and_scale_lr":
                factor = action["args"][0]["value"] if isinstance(action, dict) and action["args"] else 0.5
                if not self._has_checkpoint:
                    pass  # graceful degrade: scale LR only, matches spec §4
                else:
                    kwargs["model"].load_state_dict(self._last_checkpoint_state)
                lr_scale_factor *= factor

            elif action_name == "activate_activation_checkpointing":
                kwargs["model"].gradient_checkpointing_enable()

            elif action_name == "activate_cpu_offloading":
                pass

            elif rule.get("block"):
                for stmt in rule["block"]:
                    if stmt["call"] == "activate_activation_checkpointing":
                        kwargs["model"].gradient_checkpointing_enable()

        if lr_scale_factor != 1.0:
            for pg in kwargs["optimizer"].param_groups:
                pg["lr"] *= lr_scale_factor

        return control


class RealTorchLinearModel(torch.nn.Module):
    """A genuine torch.nn.Module — not a stand-in. Its state_dict() and
    load_state_dict() are torch's own real implementations, not a hand
    -rolled approximation of them, which is exactly the gap this tool
    exists to close relative to the earlier numpy-based verification."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(2.8)
            self.linear.bias.fill_(1.9)

    def forward(self, x):
        return self.linear(x)


class _FakeState:
    def __init__(self, step):
        self.global_step = step


class _FakeControl:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = True


def run_real_torch_guardrail_verification() -> str:
    report = []

    def log(line):
        report.append(line)

    torch.manual_seed(42)
    true_w, true_b = 3.0, 2.0
    x_all = (torch.rand(200, 1) - 0.5) * 10
    y_all = true_w * x_all + true_b + torch.randn(200, 1) * 0.1

    model = RealTorchLinearModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    callback = GuardrailCallback()

    log("PHASE 1: normal training on real torch — loss should stay quiet")
    checkpointed_weight = None
    for step in range(1, 16):
        idx = torch.randint(0, len(x_all), (16,))
        xb, yb = x_all[idx], y_all[idx]
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        control = _FakeControl()
        callback.on_log(None, _FakeState(step), control,
                         logs={"loss": loss.item()}, optimizer=optimizer, model=model)

        if step == 10:
            callback.on_save(None, _FakeState(step), _FakeControl(), model=model)
            checkpointed_weight = model.linear.weight.item()
            log(f"  step {step}: loss={loss.item():.4f}  [real on_save checkpoint taken, w={checkpointed_weight:.4f}]")
        else:
            log(f"  step {step}: loss={loss.item():.4f}")

    assert callback._has_checkpoint, "FAIL: on_save did not set _has_checkpoint"
    log("PASS: _has_checkpoint became True via a real on_save call\n")

    log("PHASE 2: real weight corruption -> real loss spike -> expect rollback_and_scale_lr")
    with torch.no_grad():
        model.linear.weight.add_(50.0)
    xb, yb = x_all[:16], y_all[:16]
    spike_loss = torch.nn.functional.mse_loss(model(xb), yb).item()
    log(f"  corrupted weight w={model.linear.weight.item():.2f} -> real loss={spike_loss:.2f}")

    lr_before = optimizer.param_groups[0]["lr"]
    callback.on_log(None, _FakeState(16), _FakeControl(),
                     logs={"loss": spike_loss}, optimizer=optimizer, model=model)
    lr_after = optimizer.param_groups[0]["lr"]
    w_after = model.linear.weight.item()

    log(f"  w after rollback: {w_after:.4f} (expected ~{checkpointed_weight:.4f})")
    log(f"  lr before: {lr_before:.5f}  lr after: {lr_after:.5f} (expected {lr_before * 0.5:.5f})")

    assert abs(w_after - checkpointed_weight) < 1e-6, "FAIL: real torch weights did not roll back"
    assert abs(lr_after - lr_before * 0.5) < 1e-9, "FAIL: LR was not scaled by 0.5"
    log("PASS: rollback_and_scale_lr genuinely restored real torch weights AND scaled LR\n")

    log("PHASE 3: flat-line loss for the configured window -> expect terminate_early")
    control = _FakeControl()
    for step in range(17, 17 + 200):
        callback.on_log(None, _FakeState(step), control,
                         logs={"loss": 0.5000001}, optimizer=optimizer, model=model)
        if control.should_training_stop:
            break
    log(f"  should_training_stop = {control.should_training_stop} (expected True)")
    assert control.should_training_stop, "FAIL: flat_lines did not trigger terminate_early"
    log("PASS: flat_lines genuinely triggered terminate_early\n")

    log("ALL PHASES PASSED on real torch.nn.Module — not numpy, not a stub.")
    return "\n".join(report)


# 2. Register tools exactly how the SDK expects them
@mcp_server.list_tools()
async def handle_list_tools():
    return [
        Tool(
            name="check_torch_env",
            description="Checks the remote server's PyTorch version and setup environment.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="verify_guardrail_callback",
            description=(
                "Runs the actual TensorScript-generated GuardrailCallback class "
                "(copied verbatim from the transpiler's output) against a real "
                "torch.nn.Module doing genuine gradient descent. Verifies "
                "checkpoint tracking via a real on_save call, genuine in-place "
                "weight rollback via real load_state_dict(), real LR scaling, "
                "and real terminate_early triggering after a flat-line loss "
                "window. Returns a step-by-step report of real (not simulated) "
                "numeric values."
            ),
            inputSchema={"type": "object", "properties": {}}
        ),
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
    if name == "verify_guardrail_callback":
        try:
            result_text = run_real_torch_guardrail_verification()
        except Exception:
            result_text = "VERIFICATION FAILED with an exception:\n\n" + traceback.format_exc()
        return [TextContent(type="text", text=result_text)]
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
