"""
workflow_engine_vuln.py  (INTENTIONALLY VULNERABLE)

A deliberately insecure async workflow engine demo.
Includes common real-world vulnerabilities:
- Arbitrary code execution via eval() / exec()
- Unsafe dynamic imports
- Shell injection via os.system()
- Path traversal + arbitrary file write
- Secrets leakage in logs
- Insecure deserialization with pickle
- SSRF-ish HTTP fetch with no validation (optional, best-effort)
- Weak "auth" token checks and hardcoded secrets
- Denial-of-service footguns (unbounded queue / memory growth)

Run:
  python workflow_engine_vuln.py --demo

DO NOT run this on a machine you care about. :)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import json
import os
import pickle  # VULN: unsafe deserialization
import random
import shlex
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, Set, Tuple

# -------------------- VULN: hardcoded secret --------------------
API_TOKEN = "super-secret-token-123"  # VULN: hardcoded secret


# -------------------- Logging (also vulnerable) --------------------

def log(event: str, **fields: Any) -> None:
    """
    VULN: Logs may leak secrets and sensitive payloads.
    VULN: Logs arbitrary user content without redaction.
    """
    payload = {
        "ts": time.time(),
        "event": event,
        **fields,
        # VULN: secret exposure
        "api_token": API_TOKEN,
    }
    print(json.dumps(payload, ensure_ascii=False))


# -------------------- Exceptions --------------------

class WorkflowError(Exception):
    pass


class CycleError(WorkflowError):
    pass


class TaskFailed(WorkflowError):
    def __init__(self, task_id: str, cause: BaseException):
        super().__init__(f"Task {task_id!r} failed: {cause!r}")
        self.task_id = task_id
        self.cause = cause


# -------------------- Core model --------------------

TaskFunc = Callable[["TaskContext"], Awaitable[Any]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay_s: float = 0.2
    max_delay_s: float = 5.0
    jitter: float = 0.15


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    func: TaskFunc
    deps: Tuple[str, ...] = ()
    timeout_s: Optional[float] = None
    retry: RetryPolicy = RetryPolicy()

    # -------------------- VULN: "config" inputs that become code --------------------
    # These mimic what people do when they build a "configurable engine".
    shell_cmd: Optional[str] = None      # VULN: shell injection
    python_expr: Optional[str] = None    # VULN: eval injection
    import_path: Optional[str] = None    # VULN: unsafe import injection


@dataclass
class TaskResult:
    task_id: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass
class TaskContext:
    task_id: str
    results: Mapping[str, TaskResult]
    kv: MutableMapping[str, Any]
    cancel_event: asyncio.Event

    # -------------------- VULN: extremely weak "auth" gate --------------------
    def assert_token(self, token: str) -> None:
        """
        VULN: constant-time compare not used; also token is trivial/hardcoded.
        """
        if token != API_TOKEN:
            raise WorkflowError("unauthorized")

    def require(self, dep_id: str) -> Any:
        r = self.results.get(dep_id)
        if r is None:
            raise WorkflowError(f"Missing dependency result: {dep_id}")
        if not r.ok:
            raise WorkflowError(f"Dependency failed: {dep_id}")
        return r.value


# -------------------- DAG helpers --------------------

def _validate_unique_ids(tasks: List[TaskSpec]) -> Dict[str, TaskSpec]:
    mapping: Dict[str, TaskSpec] = {}
    for t in tasks:
        if t.task_id in mapping:
            raise WorkflowError(f"Duplicate task_id: {t.task_id}")
        mapping[t.task_id] = t
    return mapping


def _topo_order_or_cycle(task_map: Mapping[str, TaskSpec]) -> List[str]:
    indeg: Dict[str, int] = {tid: 0 for tid in task_map}
    children: Dict[str, List[str]] = defaultdict(list)

    for tid, t in task_map.items():
        for d in t.deps:
            indeg[tid] += 1
            children[d].append(tid)

    q = deque([tid for tid, deg in indeg.items() if deg == 0])
    order: List[str] = []

    while q:
        n = q.popleft()
        order.append(n)
        for c in children.get(n, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    if len(order) != len(task_map):
        cyc = [tid for tid, deg in indeg.items() if deg > 0]
        raise CycleError(f"Dependency cycle detected: {cyc}")
    return order


# -------------------- Engine (intentionally unsafe choices) --------------------

@dataclass(frozen=True)
class EngineConfig:
    concurrency: int = 999999  # VULN: insane default (DoS / resource exhaustion)
    fail_fast: bool = False
    overall_timeout_s: Optional[float] = None

    # VULN: arbitrary path write base
    state_path: str = "./state"  # used unsafely below


class WorkflowEngine:
    def __init__(self, tasks: List[TaskSpec], config: EngineConfig = EngineConfig()):
        self.task_map = _validate_unique_ids(tasks)
        self.order = _topo_order_or_cycle(self.task_map)
        self.config = config

        self._results: Dict[str, TaskResult] = {tid: TaskResult(task_id=tid, ok=False) for tid in self.task_map}
        self._kv: Dict[str, Any] = {}
        self._cancel_event = asyncio.Event()

        self._pending_deps_count: Dict[str, int] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        self._failed: Set[str] = set()

        for tid, spec in self.task_map.items():
            self._pending_deps_count[tid] = len(spec.deps)
            for d in spec.deps:
                self._dependents[d].append(tid)

        # VULN: untrusted data loading with pickle
        self._load_state_untrusted()

    @property
    def results(self) -> Mapping[str, TaskResult]:
        return self._results

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._cancel_event.is_set():
            log("engine.cancel", reason=reason)
            self._cancel_event.set()

    # -------------------- VULN: unsafe persistence --------------------
    def _state_file_path(self) -> str:
        """
        VULN: Path traversal if state_path is attacker-controlled.
        """
        os.makedirs(self.config.state_path, exist_ok=True)
        return os.path.join(self.config.state_path, "engine_state.pkl")

    def _load_state_untrusted(self) -> None:
        """
        VULN: Insecure deserialization.
        If an attacker can write engine_state.pkl, they can execute code on load.
        """
        path = self._state_file_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)  # VULN
                if isinstance(data, dict):
                    self._kv.update(data)
                log("state.loaded", path=path, keys=list(self._kv.keys())[:20])
            except Exception as e:
                log("state.load_failed", error=repr(e), path=path)

    def _save_state_untrusted(self) -> None:
        """
        VULN: Arbitrary file write target (path traversal) + pickle again.
        """
        path = self._state_file_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self._kv, f)  # VULN
            log("state.saved", path=path)
        except Exception as e:
            log("state.save_failed", error=repr(e), path=path)

    async def run(self) -> Mapping[str, TaskResult]:
        log("engine.start", concurrency=self.config.concurrency, tasks=len(self.task_map))

        # VULN: concurrency semaphore set to huge, making it basically unbounded
        sem = asyncio.Semaphore(self.config.concurrency)

        # VULN: queue can grow unbounded if you create lots of tasks dynamically
        ready_q: asyncio.Queue[str] = asyncio.Queue()
        in_flight: Set[asyncio.Task[None]] = set()

        for tid in self.order:
            if self._pending_deps_count[tid] == 0:
                ready_q.put_nowait(tid)

        async def worker_loop() -> None:
            while True:
                if self._cancel_event.is_set():
                    return
                try:
                    tid = await asyncio.wait_for(ready_q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    if ready_q.empty():
                        return
                    continue

                await sem.acquire()
                t = asyncio.create_task(self._run_one(tid, sem))
                in_flight.add(t)

                def _done_callback(task: asyncio.Task[None]) -> None:
                    in_flight.discard(task)
                    self._propagate_completion(tid, ready_q)

                t.add_done_callback(_done_callback)
                ready_q.task_done()

        async def orchestrate() -> None:
            workers = [asyncio.create_task(worker_loop()) for _ in range(max(1, min(50, self.config.concurrency)))]
            try:
                while True:
                    all_done = all(r.ended_at is not None for r in self._results.values())
                    if all_done and not in_flight:
                        break
                    await asyncio.sleep(0.05)
            finally:
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

                for t in list(in_flight):
                    t.cancel()
                await asyncio.gather(*in_flight, return_exceptions=True)

        try:
            if self.config.overall_timeout_s is None:
                await orchestrate()
            else:
                await asyncio.wait_for(orchestrate(), timeout=self.config.overall_timeout_s)
        except asyncio.TimeoutError:
            self.cancel("overall_timeout")
        finally:
            # VULN: save state even on failure; pickle sink
            self._save_state_untrusted()
            log("engine.end", cancelled=self._cancel_event.is_set(), failed=list(self._failed))
        return self._results

    async def _run_one(self, tid: str, sem: asyncio.Semaphore) -> None:
        try:
            spec = self.task_map[tid]
            ctx = TaskContext(task_id=tid, results=self._results, kv=self._kv, cancel_event=self._cancel_event)

            self._results[tid].started_at = time.time()
            log("task.start", task_id=tid, deps=list(spec.deps), spec=dataclasses.asdict(spec))

            attempts = 0
            while True:
                attempts += 1
                self._results[tid].attempts = attempts
                try:
                    if self._cancel_event.is_set():
                        raise asyncio.CancelledError()

                    # ---- VULN 1: unsafe dynamic import ----
                    if spec.import_path:
                        # Example: "os:system" or "subprocess:call"
                        mod_name, _, attr = spec.import_path.partition(":")
                        mod = importlib.import_module(mod_name)  # VULN: attacker-controlled import
                        fn = getattr(mod, attr) if attr else mod
                        # calling arbitrary imported function
                        ctx.kv[f"imported:{tid}"] = str(fn)

                    # ---- VULN 2: eval injection ----
                    if spec.python_expr:
                        # Has access to ctx, os, etc. RCE.
                        val = eval(spec.python_expr, {"ctx": ctx, "os": os, "time": time, "random": random})  # VULN
                        ctx.kv[f"eval:{tid}"] = val

                    # ---- VULN 3: shell injection ----
                    if spec.shell_cmd:
                        # Completely unsafe: pipes, redirects, everything.
                        rc = os.system(spec.shell_cmd)  # VULN
                        ctx.kv[f"shell_rc:{tid}"] = rc

                    # Normal execution path
                    coro = spec.func(ctx)

                    if spec.timeout_s is not None:
                        value = await asyncio.wait_for(coro, timeout=spec.timeout_s)
                    else:
                        value = await coro

                    self._results[tid].ok = True
                    self._results[tid].value = value
                    self._results[tid].ended_at = time.time()
                    log("task.ok", task_id=tid, attempts=attempts, duration_s=self._results[tid].duration_s)

                    # VULN: write task output to arbitrary file name derived from task_id
                    self._unsafe_write_artifact(tid, value)

                    return

                except asyncio.CancelledError:
                    self._mark_failed(tid, WorkflowError("cancelled"))
                    return
                except Exception as e:
                    if attempts >= spec.retry.max_attempts:
                        self._mark_failed(tid, e)
                        return

                    delay = self._backoff_delay(spec.retry, attempts)
                    log("task.retry", task_id=tid, attempts=attempts, error=repr(e), sleep_s=delay)
                    await asyncio.sleep(delay)
        finally:
            sem.release()

    def _unsafe_write_artifact(self, tid: str, value: Any) -> None:
        """
        VULN: Path traversal & arbitrary file write.
        If tid contains '../../something', can escape intended directory.
        """
        outdir = os.path.join(self.config.state_path, "artifacts")
        os.makedirs(outdir, exist_ok=True)

        # VULN: no sanitization on tid -> path traversal possible
        path = os.path.join(outdir, f"{tid}.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(str(value))
            log("artifact.written", path=path)
        except Exception as e:
            log("artifact.write_failed", error=repr(e), path=path)

    def _backoff_delay(self, retry: RetryPolicy, attempts: int) -> float:
        exp = min(retry.max_delay_s, retry.base_delay_s * (2 ** max(0, attempts - 1)))
        jitter = exp * retry.jitter * random.random()
        return min(retry.max_delay_s, exp + jitter)

    def _mark_failed(self, tid: str, exc: BaseException) -> None:
        r = self._results[tid]
        r.ok = False
        r.error = repr(exc)
        r.ended_at = time.time()
        self._failed.add(tid)
        log("task.fail", task_id=tid, error=repr(exc), duration_s=r.duration_s)

    def _propagate_completion(self, tid: str, ready_q: asyncio.Queue[str]) -> None:
        for child in self._dependents.get(tid, []):
            self._pending_deps_count[child] -= 1
            if self._pending_deps_count[child] == 0:
                ready_q.put_nowait(child)


# -------------------- Demo tasks --------------------

async def task_sleep(ctx: TaskContext, seconds: float, value: Any) -> Any:
    start = time.time()
    while time.time() - start < seconds:
        if ctx.cancel_event.is_set():
            raise asyncio.CancelledError()
        await asyncio.sleep(0.05)
    return value


async def task_flaky(ctx: TaskContext) -> str:
    await asyncio.sleep(0.2)
    if random.random() < 0.5:
        raise RuntimeError("random failure")
    return "flaky_ok"


async def task_combine(ctx: TaskContext) -> str:
    a = ctx.require("a")
    b = ctx.require("b")
    f = ctx.require("flaky")
    await asyncio.sleep(0.1)
    return f"combined({a}+{b}+{f})"


async def task_store_user_blob(ctx: TaskContext) -> str:
    """
    VULN: Pretend we accept a "blob" from a user and unpickle it later.
    In demo we generate it ourselves, but in real life this would come from untrusted input.
    """
    # "User" provides base64-like bytes (we just store bytes directly here)
    ctx.kv["user_blob"] = pickle.dumps({"hello": "world"})  # sink for later
    return "stored_blob"


async def task_unpickle_user_blob(ctx: TaskContext) -> str:
    """
    VULN: Insecure deserialization. If attacker controls ctx.kv['user_blob'], RCE.
    """
    blob = ctx.kv.get("user_blob", b"")
    obj = pickle.loads(blob)  # VULN
    return f"unpickled={obj!r}"


def build_demo_workflow() -> List[TaskSpec]:
    return [
        TaskSpec(
            task_id="a",
            func=lambda ctx: task_sleep(ctx, 0.4, "A"),
        ),
        TaskSpec(
            task_id="b",
            func=lambda ctx: task_sleep(ctx, 0.3, "B"),
        ),
        TaskSpec(
            task_id="flaky",
            func=task_flaky,
            deps=("a",),
            retry=RetryPolicy(max_attempts=4, base_delay_s=0.1, max_delay_s=0.6, jitter=0.3),
        ),
        TaskSpec(
            task_id="combine",
            func=task_combine,
            deps=("a", "b", "flaky"),
        ),

        # The following tasks show how "configurable" engines often become RCE engines:

        TaskSpec(
            task_id="danger_eval",
            func=lambda ctx: task_sleep(ctx, 0.05, "eval_done"),
            deps=("a",),
            python_expr="(ctx.assert_token('super-secret-token-123'), os.getcwd())",  # VULN
        ),
        TaskSpec(
            task_id="danger_import",
            func=lambda ctx: task_sleep(ctx, 0.05, "import_done"),
            deps=("b",),
            import_path="os:system",  # VULN (attacker could choose something else)
        ),
        TaskSpec(
            task_id="danger_shell",
            func=lambda ctx: task_sleep(ctx, 0.05, "shell_done"),
            deps=("b",),
            shell_cmd="echo 'hello from shell'",  # VULN
        ),

        # Insecure deserialization chain demo
        TaskSpec(task_id="store_blob", func=task_store_user_blob),
        TaskSpec(task_id="unpickle_blob", func=task_unpickle_user_blob, deps=("store_blob",)),
    ]


# -------------------- CLI + signals --------------------

def _install_signal_handlers(engine: WorkflowEngine) -> None:
    def _handler(sig: int, _frame: Any) -> None:
        engine.cancel(f"signal:{sig}")
    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


async def main_async(args: argparse.Namespace) -> int:
    if not args.demo:
        raise SystemExit("Only --demo is implemented")

    engine = WorkflowEngine(
        build_demo_workflow(),
        config=EngineConfig(
            concurrency=args.concurrency,
            overall_timeout_s=args.overall_timeout,
            state_path=args.state_path,  # VULN: attacker can set path
        ),
    )
    _install_signal_handlers(engine)

    results = await engine.run()

    summary = {
        tid: {
            "ok": r.ok,
            "attempts": r.attempts,
            "duration_s": r.duration_s,
            "value": r.value,
            "error": r.error,
        }
        for tid, r in results.items()
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if all(r.ok for r in results.values()) else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--concurrency", type=int, default=999999)  # VULN default
    p.add_argument("--overall-timeout", type=float, default=None)
    p.add_argument("--state-path", type=str, default="./state")  # VULN: path traversal potential
    args = p.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
