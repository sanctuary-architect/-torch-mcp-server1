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
# GuardrailCallback, copied verbatim from the TensorScript transpiler's
# output for `optimize BaseLM as SFTStage`. Unchanged from the version
# already verified against a real torch.nn.Module in the numpy-free toy
# test. The only adaptation is the base class, since this server doesn't
# carry the full transformers/peft/datasets stack for check_torch_env's
# sake — but see below, this file now DOES have transformers, so this
# subclasses the real transformers.TrainerCallback for the first time.
from transformers import TrainerCallback, TrainingArguments, Trainer, GPT2Config, GPT2LMHeadModel


class GuardrailCallback(TrainerCallback):
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


# ---------------------------------------------------------------------------
# The original manual-driven verification tool (no real Trainer involved —
# GuardrailCallback's on_log/on_save are called directly, not dispatched by
# Trainer internals). Kept alongside the newer Trainer-based tool below so
# both verification levels stay available.

class RealTorchLinearModel(torch.nn.Module):
    """A genuine torch.nn.Module — not a stand-in. Its state_dict() and
    load_state_dict() are torch's own real implementations, not a hand
    -rolled approximation of them."""

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


# ---------------------------------------------------------------------------
# New: a genuine transformers.Trainer integration test. Everything above
# this point (GuardrailCallback) has now been checked under three
# progressively more real conditions: hand-fed fake state, a real torch
# linear model driven manually, and now a real Trainer loop driving a
# real (tiny, from-scratch, no-download) GPT-2 causal LM.

class TinyLMDataset(torch.utils.data.Dataset):
    """Synthetic token sequences — no tokenizer, no download. The model
    only needs to see *some* real gradient signal; what the tokens mean
    is irrelevant to what this test verifies."""

    def __init__(self, vocab_size, seq_len, n_examples, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_examples, seq_len), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids.clone()}


class ChaosInjectionCallback(TrainerCallback):
    """Deterministically corrupts one real parameter tensor mid-run, the
    same role the manual 'model.linear.weight.add_(50.0)' line played in
    the earlier torch-only test — except this time it's triggered by a
    genuine Trainer step boundary (on_step_end), not called by hand."""

    def __init__(self, inject_at_step):
        self.inject_at_step = inject_at_step
        self.injected = False
        self.corrupted_value = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.inject_at_step and not self.injected:
            model = kwargs["model"]
            with torch.no_grad():
                first_param = next(model.parameters())
                first_param.add_(50.0)
                self.corrupted_value = first_param.flatten()[0].item()
            self.injected = True
        return control


class ReportingCallback(TrainerCallback):
    """Placed AFTER GuardrailCallback in the callbacks list. Trainer's own
    callback dispatch (call_event) iterates callbacks in list order for a
    given event, so this on_log always runs strictly after
    GuardrailCallback's on_log for the same event — letting it observe
    whatever GuardrailCallback just did (including a rollback) without
    needing to modify GuardrailCallback itself."""

    def __init__(self, tracked_param_getter):
        self.tracked_param_getter = tracked_param_getter
        self.events = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.events.append({
                "step": state.global_step,
                "loss": logs["loss"],
                "tracked_param": self.tracked_param_getter(kwargs["model"]),
                "should_training_stop": control.should_training_stop,
            })
        return control


def run_real_trainer_guardrail_verification() -> str:
    report = []

    def log(line):
        report.append(line)

    torch.manual_seed(7)
    vocab_size, seq_len = 50, 8
    config = GPT2Config(
        vocab_size=vocab_size, n_positions=seq_len,
        n_embd=16, n_layer=2, n_head=2,
    )
    model = GPT2LMHeadModel(config)
    dataset = TinyLMDataset(vocab_size=vocab_size, seq_len=seq_len, n_examples=32)

    def get_tracked_param(m):
        return next(m.parameters()).flatten()[0].item()

    checkpoint_value_holder = {}

    guardrail_cb = GuardrailCallback()
    chaos_cb = ChaosInjectionCallback(inject_at_step=3)
    reporting_cb = ReportingCallback(tracked_param_getter=get_tracked_param)

    training_args = TrainingArguments(
        output_dir="/tmp/tensorscript_real_trainer_verification",
        max_steps=6,
        per_device_train_batch_size=4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=2,
        save_total_limit=1,
        report_to=[],
        disable_tqdm=True,
        learning_rate=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # Order matters: chaos injects at on_step_end (before this step's
        # on_log fires), guardrail reacts to the resulting logged loss,
        # reporting observes the aftermath — all via real Trainer dispatch.
        callbacks=[chaos_cb, guardrail_cb, reporting_cb],
    )

    log("Running a genuine transformers.Trainer loop — real GPT-2 architecture,")
    log("real optimizer, real checkpoint saving, real callback dispatch.\n")

    trainer.train()

    log(f"{'step':>4}  {'loss':>10}  {'tracked_param':>14}  should_stop")
    for e in reporting_cb.events:
        log(f"{e['step']:>4}  {e['loss']:>10.4f}  {e['tracked_param']:>14.4f}  {e['should_training_stop']}")

    assert guardrail_cb._has_checkpoint, "FAIL: GuardrailCallback.on_save was never triggered by real Trainer"
    log("\nPASS: on_save fired from genuine Trainer checkpoint saving (save_steps=2)")

    assert chaos_cb.injected, "FAIL: chaos injection never ran"
    corrupted = chaos_cb.corrupted_value
    log(f"\nChaos injected a real corrupted param value: {corrupted:.4f}")

    # Find the step immediately after injection and confirm the tracked
    # param no longer matches the corrupted value — i.e. it was rolled back.
    post_injection_events = [e for e in reporting_cb.events if e["step"] >= chaos_cb.inject_at_step]
    assert post_injection_events, "FAIL: no logged events after chaos injection"
    post_value = post_injection_events[0]["tracked_param"]
    log(f"Tracked param immediately after: {post_value:.4f}")

    assert abs(post_value - corrupted) > 1.0, (
        "FAIL: tracked param still matches the corrupted value — "
        "rollback_and_scale_lr did not actually revert real Trainer-managed weights"
    )
    log("PASS: real Trainer-managed weights were genuinely rolled back — not just logged")

    log("\nALL CHECKS PASSED under genuine transformers.Trainer orchestration.")
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
                "against a real torch.nn.Module doing manually-driven gradient "
                "descent. Verifies checkpoint tracking, in-place weight "
                "rollback, LR scaling, and terminate_early — all hand-fed, not "
                "under a real Trainer loop."
            ),
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="verify_guardrail_with_real_trainer",
            description=(
                "Runs the actual TensorScript-generated GuardrailCallback class "
                "copied verbatim, this time attached to a genuine "
                "transformers.Trainer training a real (tiny, from-scratch) "
                "GPT-2 model. Verifies on_save and on_log are correctly "
                "triggered by real Trainer internals (not simulated), and "
                "that rollback_and_scale_lr genuinely reverts real "
                "Trainer-managed model weights after a deterministically "
                "injected loss spike."
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
    if name == "verify_guardrail_with_real_trainer":
        try:
            result_text = run_real_trainer_guardrail_verification()
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

# 5. Mount the MCP transport as raw Starlette Route/Mount objects — see
#    the note in handle_sse's docstring and prior debugging history for
#    why this can't be FastAPI's @app.get/@app.post decorators.
app.router.routes.append(Route("/sse", endpoint=handle_sse, methods=["GET"]))
app.router.routes.append(Mount("/messages/", app=sse_transport.handle_post_message))
