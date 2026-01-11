"""
workflow_engine.py

A compact-but-difficult async workflow engine:
- Define tasks with dependencies (DAG)
- Runs tasks concurrently with a global concurrency limit
- Per-task retries with exponential backoff + jitter
- Per-task timeout
- Cancellation propagation + graceful shutdown
- Cycle detection + topological scheduling
- Structured JSON logging

Requires: Python 3.11+
Usage:
  python workflow_engine.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
)

# ---------- Structured logging ----------

def log(event: str, **fields: Any) -> None:
    payload = {
        "ts": time.time(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False))


# ---------- Exceptions ----------

class WorkflowError(Exception):
    pass


class CycleError(WorkflowError):
    pass


class TaskFailed(WorkflowError):
    def __init__(self, task_id: str, cause: BaseException):
        super().__init__(f"Task {task_id!r} failed: {cause!r}")
        self.task_id = task_id
        self.cause = cause


class DependencyFailed(WorkflowError):
    def __init__(self, task_id: str, dep_id: str):
        super().__init__(f"Task {task_id!r} blocked: dependency {dep_id!r} failed")
        self.task_id = task_id
        self.dep_id = dep_id


# ---------- Core model ----------

TaskFunc = Callable[["TaskContext"], Awaitable[Any]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    base_delay_s: float = 0.2
    max_delay_s: float = 5.0
    jitter: float = 0.15  # 0..1 fraction


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    func: TaskFunc
    deps: Tuple[str, ...] = ()
    timeout_s: Optional[float] = None
    retry: RetryPolicy = RetryPolicy()


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
    """What tasks can see while running."""
    task_id: str
    results: Mapping[str, TaskResult]
    kv: MutableMapping[str, Any]
    cancel_event: asyncio.Event

    def require(self, dep_id: str) -> Any:
        r = self.results.get(dep_id)
        if r is None:
            raise WorkflowError(f"Missing dependency result: {dep_id}")
        if not r.ok:
            raise DependencyFailed(self.task_id, dep_id)
        return r.value


# ---------- DAG helpers ----------

def _validate_unique_ids(tasks: Iterable[TaskSpec]) -> Dict[str, TaskSpec]:
    mapping: Dict[str, TaskSpec] = {}
    for t in tasks:
        if t.task_id in mapping:
            raise WorkflowError(f"Duplicate task_id: {t.task_id}")
        mapping[t.task_id] = t
    return mapping


def _validate_deps_exist(task_map: Mapping[str, TaskSpec]) -> None:
    for t in task_map.values():
        for d in t.deps:
            if d not in task_map:
                raise WorkflowError(f"Task {t.task_id!r} depends on unknown task {d!r}")


def _topo_order_or_cycle(task_map: Mapping[str, TaskSpec]) -> List[str]:
    """Kahn's algorithm. Raises CycleError if cycle exists."""
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
        # Find a cycle-ish set: nodes with indeg>0
        cyc = [tid for tid, deg in indeg.items() if deg > 0]
        raise CycleError(f"Dependency cycle detected involving: {cyc}")
    return order


# ---------- Engine ----------

@dataclass(frozen=True)
class EngineConfig:
    concurrency: int = 4
    fail_fast: bool = True  # cancel remaining runnable tasks on first failure
    overall_timeout_s: Optional[float] = None


class WorkflowEngine:
    def __init__(self, tasks: Iterable[TaskSpec], config: EngineConfig = EngineConfig()):
        self.task_map = _validate_unique_ids(tasks)
        _validate_deps_exist(self.task_map)
        self.order = _topo_order_or_cycle(self.task_map)
        self.config = config

        # runtime state
        self._results: Dict[str, TaskResult] = {tid: TaskResult(task_id=tid, ok=False) for tid in self.task_map}
        self._kv: Dict[str, Any] = {}
        self._cancel_event = asyncio.Event()

        # scheduling helpers
        self._pending_deps_count: Dict[str, int] = {}
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        self._failed: Set[str] = set()

        for tid, spec in self.task_map.items():
            self._pending_deps_count[tid] = len(spec.deps)
            for d in spec.deps:
                self._dependents[d].append(tid)

    @property
    def results(self) -> Mapping[str, TaskResult]:
        return self._results

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._cancel_event.is_set():
            log("engine.cancel", reason=reason)
            self._cancel_event.set()

    async def run(self) -> Mapping[str, TaskResult]:
        log("engine.start", concurrency=self.config.concurrency, fail_fast=self.config.fail_fast)

        sem = asyncio.Semaphore(self.config.concurrency)
        ready_q: asyncio.Queue[str] = asyncio.Queue()
        in_flight: Set[asyncio.Task[None]] = set()

        # seed initial ready tasks
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
                    # exit when no tasks left and none in flight
                    if ready_q.empty():
                        return
                    continue

                # If already done (can happen in fail-fast cascades), skip.
                if self._results[tid].ended_at is not None:
                    ready_q.task_done()
                    continue

                # If any dependency failed, mark blocked and propagate.
                spec = self.task_map[tid]
                failed_dep = next((d for d in spec.deps if d in self._failed), None)
                if failed_dep is not None:
                    self._mark_failed(tid, DependencyFailed(tid, failed_dep))
                    ready_q.task_done()
                    self._propagate_completion(tid, ready_q)
                    continue

                await sem.acquire()
                t = asyncio.create_task(self._run_one(tid, sem))
                in_flight.add(t)

                def _done_callback(task: asyncio.Task[None]) -> None:
                    in_flight.discard(task)
                    # Propagate completions and unlock dependents
                    self._propagate_completion(tid, ready_q)

                t.add_done_callback(_done_callback)
                ready_q.task_done()

                if self.config.fail_fast and self._failed:
                    self.cancel("fail_fast: a task failed")
                    return

        async def orchestrate() -> None:
            # Run a few worker loops concurrently to keep queue draining.
            workers = [asyncio.create_task(worker_loop()) for _ in range(max(1, self.config.concurrency))]
            try:
                while True:
                    if self._cancel_event.is_set():
                        break

                    # if everything done, break
                    all_done = all(r.ended_at is not None for r in self._results.values())
                    if all_done and not in_flight:
                        break

                    await asyncio.sleep(0.05)
            finally:
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

                # ensure in-flight tasks are cancelled on shutdown
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
            log("engine.end", cancelled=self._cancel_event.is_set(), failed_count=len(self._failed))
        return self._results

    async def _run_one(self, tid: str, sem: asyncio.Semaphore) -> None:
        try:
            spec = self.task_map[tid]
            ctx = TaskContext(
                task_id=tid,
                results=self._results,
                kv=self._kv,
                cancel_event=self._cancel_event,
            )

            self._results[tid].started_at = time.time()
            log("task.start", task_id=tid, deps=list(spec.deps))

            attempts = 0
            while True:
                attempts += 1
                self._results[tid].attempts = attempts
                try:
                    if self._cancel_event.is_set():
                        raise asyncio.CancelledError()

                    coro = spec.func(ctx)
                    if spec.timeout_s is not None:
                        value = await asyncio.wait_for(coro, timeout=spec.timeout_s)
                    else:
                        value = await coro

                    self._results[tid].ok = True
                    self._results[tid].value = value
                    self._results[tid].ended_at = time.time()
                    log("task.ok", task_id=tid, attempts=attempts, duration_s=self._results[tid].duration_s)
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

    def _backoff_delay(self, retry: RetryPolicy, attempts: int) -> float:
        # attempts starts at 1
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
        # decrease pending deps for dependents, enqueue if now ready
        for child in self._dependents.get(tid, []):
            self._pending_deps_count[child] -= 1
            if self._pending_deps_count[child] == 0:
                ready_q.put_nowait(child)


# ---------- Demo tasks ----------

async def task_sleep(ctx: TaskContext, seconds: float, value: Any) -> Any:
    # cooperative cancellation
    start = time.time()
    while time.time() - start < seconds:
        if ctx.cancel_event.is_set():
            raise asyncio.CancelledError()
        await asyncio.sleep(0.05)
    return value


async def task_flaky(ctx: TaskContext) -> str:
    # fails ~50% until it succeeds
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


async def task_timeout(ctx: TaskContext) -> str:
    # intentionally long
    await asyncio.sleep(10)
    return "should_not_happen"


def build_demo_workflow() -> List[TaskSpec]:
    return [
        TaskSpec(
            task_id="a",
            func=lambda ctx: task_sleep(ctx, 0.6, "A"),
            deps=(),
            retry=RetryPolicy(max_attempts=1),
        ),
        TaskSpec(
            task_id="b",
            func=lambda ctx: task_sleep(ctx, 0.4, "B"),
            deps=(),
            retry=RetryPolicy(max_attempts=1),
        ),
        TaskSpec(
            task_id="flaky",
            func=task_flaky,
            deps=("a",),
            retry=RetryPolicy(max_attempts=5, base_delay_s=0.1, max_delay_s=1.0, jitter=0.4),
        ),
        TaskSpec(
            task_id="combine",
            func=task_combine,
            deps=("a", "b", "flaky"),
            retry=RetryPolicy(max_attempts=1),
        ),
        TaskSpec(
            task_id="timeouty",
            func=task_timeout,
            deps=("b",),
            timeout_s=0.5,
            retry=RetryPolicy(max_attempts=2, base_delay_s=0.1, max_delay_s=0.3, jitter=0.2),
        ),
    ]


# ---------- CLI ----------

def _install_signal_handlers(engine: WorkflowEngine) -> None:
    # Works on Unix well; on Windows, SIGINT is OK but SIGTERM may vary.
    def _handler(sig: int, _frame: Any) -> None:
        engine.cancel(f"signal:{sig}")

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        # some environments disallow installing signal handlers
        pass


async def main_async(args: argparse.Namespace) -> int:
    if args.demo:
        tasks = build_demo_workflow()
    else:
        raise SystemExit("Only --demo is implemented in this single-file example.")

    engine = WorkflowEngine(
        tasks,
        config=EngineConfig(
            concurrency=args.concurrency,
            fail_fast=args.fail_fast,
            overall_timeout_s=args.overall_timeout,
        ),
    )
    _install_signal_handlers(engine)

    results = await engine.run()

    # Pretty summary (still JSON-ish)
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

    # Return non-zero if any failed
    return 0 if all(r.ok for r in results.values()) else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Async DAG workflow engine demo")
    p.add_argument("--demo", action="store_true", help="run the demo workflow")
    p.add_argument("--concurrency", type=int, default=4, help="max concurrent tasks")
    p.add_argument("--fail-fast", action="store_true", help="cancel remaining tasks on first failure")
    p.add_argument("--overall-timeout", type=float, default=None, help="overall workflow timeout (seconds)")
    args = p.parse_args()

    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
