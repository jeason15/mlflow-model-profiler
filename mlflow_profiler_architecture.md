# `mlflow-profiler` — Architecture Proposal

**Version:** 0.2  
**Status:** Proposal  
**Last Updated:** 2026-02-18

---

## 1. Problem

ML engineers diagnosing inference latency in MLflow models face a gap in
available tooling. The existing options each fail in a specific way:

**Ad-hoc timing** (`time.perf_counter()`) requires the model author to
refactor monolithic `predict()` methods into small functions and manually
instrument each one. This doesn't scale — when you're parachuting into a
team's model to diagnose a latency issue, you can't ask them to restructure
their code first.

**MLflow Tracing** is architecturally wrong for traditional ML. It's built
on OpenTelemetry for GenAI call-chain observability — serializing spans as
JSON to a relational store, with enough overhead that Databricks recommends
async logging. The dependency footprint alone (OpenTelemetry + protobuf +
grpc) is disqualifying for lightweight inference containers.

**`cProfile`** hooks every function call and produces a flat statistical
table with no call-tree context. The ~3x overhead makes it unsafe for
anything resembling production traffic, and the output format requires
significant post-processing to extract actionable insight.

**`pyinstrument`** samples at 1ms intervals — too coarse to resolve the
sub-millisecond spans typical in ML preprocessing pipelines. It also adds
a C extension dependency and provides no visibility into data shapes or
types flowing through the pipeline.

**The gap:** There is no tool that can automatically capture a structured
call tree from an uninstrumented model's `predict()` call, at a
user-controlled depth, with negligible overhead at sensible defaults, zero
required dependencies, and optional integration with MLflow's tracking
ecosystem.

### Vision

`mlflow-profiler` is an observability layer for ML model inference. It
occupies the same role for traditional ML that MLflow Tracing occupies for
GenAI — but built from scratch for the latency and overhead constraints of
real-time model serving.

The long-term trajectory:

1. Standalone profiling tool for interactive diagnosis
2. First-class PyPI package with stable API
3. Online monitoring with I/O tracking and UI integration
4. Candidate for contribution to the MLflow ecosystem

---

## 2. Core Design Decisions

Each decision below was evaluated against alternatives. The rationale
sections explain why we chose what we chose.

### 2.1 Capture Mechanism: `sys.setprofile` with Depth Ceiling

**Decision:** Use Python's `sys.setprofile()` hook to capture function
call/return events, governed by an integer `depth` parameter that acts
as a ceiling on how deep into the call tree we record.

**Alternatives considered:**

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Manual decorators only (current approach) | Zero overhead when absent, precise | Requires model refactoring; non-starter for uninstrumented models | Retained as optional layer |
| `sys.setprofile` with filtering heuristics | No model changes needed | Heuristics are fragile; "what's interesting" is subjective | Rejected in favor of depth ceiling |
| `sys.setprofile` with depth ceiling | No model changes; user controls granularity; capture once, render at any level | Per-event overhead on hot path | **Selected** |
| `sys.settrace` | More events (line-level) | Much higher overhead; overkill for function-level profiling | Rejected |
| Bytecode rewriting / AST transform | Could inject timing at arbitrary points | Fragile, version-dependent, unsafe | Rejected |

**How it works:**

The profiler installs a callback via `sys.setprofile()` for the duration
of a profiling session. The callback fires on every function `call` and
`return` event (and optionally `c_call`/`c_return` for C-extension
functions). An integer depth counter tracks the current position in the
call stack. Events deeper than the ceiling are skipped with a single
integer comparison.

```
depth=0   →  Just the root predict() span
depth=1   →  predict + its direct callees
depth=2   →  Two levels deep (recommended default)
depth=-1  →  Unlimited — capture every frame
```

**The key property: capture once, render at any level.** A session captured
at `depth=5` can be rendered at depth 0, 1, 2, 3, 4, or 5 after the fact.
Nodes below the render depth are not shown; their time is already included
in the parent's wall-clock interval, so aggregation is simply "hide the
deeper nodes." No recalculation required.

**Overhead model:**

Per-event cost when frame IS within the ceiling:
- `time.perf_counter_ns()`: ~30ns
- Tuple creation + list append: ~40-70ns
- Total: ~100-150ns

Per-event cost when frame is BELOW the ceiling:
- Integer increment + comparison: ~10-20ns

At `depth=2` on a typical model predict, we capture ~10-40 events. Total
overhead: ~2-6μs. On a 10ms predict call, that's 0.02-0.06%.

### 2.2 Context Propagation: `ContextVar`, not `threading.local`

**Decision:** Use `contextvars.ContextVar` to hold the active
`ProfileSession` reference.

**Rationale:** The original module uses `threading.local()`, which fails
silently in async contexts (`asyncio`, `trio`) and in thread pools that
use `contextvars.copy_context()`. `ContextVar` is the mechanism MLflow
Tracing itself uses for span context propagation. It's implemented in C
in CPython and has equivalent performance to `threading.local` attribute
access (~30-50ns for `.get()`).

**Implication:** When no session is active, `_current_session.get(None)`
returns `None` in ~30-50ns. This is the ONLY overhead for disabled
decorators — there is no flag check, no dictionary lookup, no session ID
resolution.

```python
_current_session: ContextVar[ProfileSession | None] = ContextVar(
    '_current_session', default=None
)
```

### 2.3 Span Identity: Integer Index, not UUID

**Decision:** Spans are identified by their integer index in the session's
span list. UUIDs are not generated during capture.

**Rationale:** `uuid.uuid4()` costs ~1-2μs per call (involves
`os.urandom(16)` + formatting). An integer index costs nothing — it's just
`len(self.spans)` at the time of creation. For the single-process,
single-session scope of a profile, indices are unambiguous.

If globally unique IDs are needed for MLflow persistence, they are generated
lazily at export time — off the hot path.

### 2.4 Event Buffer: Flat List, Lazy Tree

**Decision:** During capture, events are appended to a flat list of tuples.
Tree reconstruction happens lazily on first access, after profiling is
complete.

**Rationale:** Building a tree during capture requires parent lookups,
node creation, and potentially dictionary operations on every frame event.
A flat append to a Python list is the cheapest possible capture operation.
Tree reconstruction is O(n) in the number of events and happens once, off
the hot path, where latency is irrelevant.

```python
# During capture (hot path): just append
self._events.append((timestamp_ns, event_type, code_ref))

# After capture (cold path): reconstruct tree
self._spans = _build_tree(self._events)
```

### 2.5 Patch Target: `PyFuncModel.predict`, not `PythonModel.predict`

**Decision:** The `autoprofile()` mechanism patches MLflow's `PyFuncModel`
wrapper class, not the user's `PythonModel` subclass.

**Rationale:** `PyFuncModel` is the object returned by
`mlflow.pyfunc.load_model()`. It's MLflow's own delegation wrapper — the
same level at which MLflow's autologging operates. Patching it means:

- We never touch user class definitions or instances
- We never modify the model's `__dict__` or MRO
- The patch point is stable across MLflow versions (it's a public API surface)
- Unpatching restores the exact original method reference

The alternative — patching `PythonModel.predict` or individual subclass
methods — is more invasive, harder to reason about with inheritance, and
risks interfering with MLflow's own `__init_subclass__` wrapping of predict.

### 2.6 I/O Capture: Lightweight Summaries, Never References

**Decision:** Input/output data is immediately reduced to an `IOSummary`
dataclass containing only primitive metadata (type name, shape, dtype,
size, device, short repr). No references to user objects are ever stored.

**Rationale:** Holding references to intermediate tensors could pin GPU
memory. Holding references to DataFrames could prevent garbage collection
of large allocations. Serializing full objects is prohibitively expensive.

`IOSummary` captures exactly the metadata that's diagnostically useful:

```python
@dataclasses.dataclass(slots=True, frozen=True)
class IOSummary:
    type_name: str              # "pandas.core.frame.DataFrame"
    shape: tuple | None         # (1000, 50)
    dtype: str | None           # "float32"
    length: int | None          # 1000
    size_bytes: int | None      # 400000
    device: str | None          # "cuda:0" (torch tensors)
    repr_short: str | None      # First 80 chars for scalars/strings
```

Generating this from an object costs ~200-500ns (a few `getattr` calls
and one `sys.getsizeof`). The original object is never stored and is
eligible for GC immediately.

For auto-captured frames, input summarization requires accessing
`frame.f_locals`, which creates a dict copy (~500ns-1μs). This cost is
only incurred for frames within the depth ceiling, so it scales with
`depth`, not with total call count.

### 2.7 Three-Layer API

**Decision:** The API is structured in three layers, from simplest to most
flexible. All three compose — they share the same `ContextVar`-based
session and produce the same `SpanRecord` data.

**Layer 1: `autoprofile()`** — One-liner. Patches `PyFuncModel.predict()`
to wrap calls in a profiling session automatically. Zero model changes.
This is the entry point for "I just want to see what's happening."

**Layer 2: `profiling()` context manager + `@profile_span` decorator** —
For ad-hoc sessions and for model authors who want to label specific
methods. Decorators integrate into the auto-captured tree when a session
is active; they are complete no-ops when no session is active.

**Layer 3: `ProfileSession` direct** — For integration authors, custom
tooling, and programmatic access to raw span data.

The layering means a user's first experience is:

```python
import mlflow_profiler
mlflow_profiler.autoprofile()
model.predict(data)
mlflow_profiler.last_profile().print_tree()
```

And they can progressively add specificity as needed, without changing
their approach — just adding `@profile_span` labels or switching from
`autoprofile()` to explicit `with profiling()` blocks.

---

## 3. Data Model

### 3.1 SpanRecord

The atomic unit of profile data. One per captured function call.

```python
@dataclasses.dataclass(slots=True)
class SpanRecord:
    label: str                          # Function qualified name or custom label
    module: str | None                  # Module where function is defined
    start_ns: int                       # time.perf_counter_ns() at entry
    end_ns: int | None = None           # time.perf_counter_ns() at exit
    parent_index: int | None = None     # Index into session.spans
    depth: int = 0                      # Call depth (0 = root)
    input_summary: dict[str, IOSummary] | None = None  # Arg name → summary
    output_summary: IOSummary | None = None             # Return value summary
    is_user_code: bool = False          # True if defined outside site-packages

    @property
    def duration_ns(self) -> int | None:
        if self.end_ns is None:
            return None
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float | None:
        d = self.duration_ns
        return d / 1_000_000 if d is not None else None
```

### 3.2 IOSummary

Lightweight, serialization-safe descriptor of a data object.

```python
@dataclasses.dataclass(slots=True, frozen=True)
class IOSummary:
    type_name: str
    shape: tuple | None = None
    dtype: str | None = None
    length: int | None = None
    size_bytes: int | None = None
    device: str | None = None
    repr_short: str | None = None
```

The `summarize()` function extracts this from any Python object via
guarded `getattr` calls. It handles the common ML types natively:

- `numpy.ndarray`: shape, dtype, nbytes
- `pandas.DataFrame` / `Series`: shape, dtypes, memory_usage
- `torch.Tensor`: shape, dtype, device, nbytes
- `dict` / `list`: length, getsizeof
- Scalars: repr

Custom types are handled gracefully (type name + repr_short as fallback).
A `register_summarizer()` extension point allows users to register custom
extraction logic for domain-specific types.

### 3.3 ProfileSession

Owns the capture lifecycle and provides all rendering/export methods.

```python
class ProfileSession:
    def __init__(self, depth=2, capture_io=True): ...

    # ── Lifecycle ──────────────────────────────────────
    def activate(self) -> Token: ...      # Install setprofile, set ContextVar
    def deactivate(self, token): ...      # Remove setprofile, reset ContextVar

    # ── Data Access ────────────────────────────────────
    @property
    def spans(self) -> list[SpanRecord]:  # Lazy tree build on first access
        ...

    # ── Rendering ──────────────────────────────────────
    def print_tree(self, depth=None, show_io=True,
                   collapse_frameworks=False): ...
    def to_tree(self, depth=None) -> list[dict]: ...
    def to_flat(self, depth=None) -> list[dict]: ...
    def to_json(self, depth=None) -> str: ...
    def to_chrome_trace(self) -> str: ...
```

---

## 4. Render Modes

All rendering is post-hoc. The same captured session supports multiple views.

### Depth-Based Aggregation

```
# Captured at depth=3, rendered at depth=1
predict: 48.12ms
  preprocess: 2.31ms
  _run_model: 44.50ms
  postprocess: 0.03ms

# Same session, rendered at depth=2
predict: 48.12ms
  preprocess: 2.31ms
    tokenizer.encode: 1.87ms
    numpy.stack: 0.12ms
  _run_model: 44.50ms
    Linear.forward: 22.10ms
    Linear.forward: 22.30ms
  postprocess: 0.03ms

# Same session, rendered at depth=3
predict: 48.12ms
  preprocess: 2.31ms
    tokenizer.encode: 1.87ms
      _byte_pair_encode: 1.60ms
      _convert_tokens: 0.25ms
    numpy.stack: 0.12ms
  ...
```

### I/O Display

```
predict: 48.12ms
  in:  (DataFrame, shape=(1000, 50), dtype=float64, 400.0KB)
  out: (ndarray, shape=(1000,), dtype=int64, 7.8KB)

  preprocess: 2.31ms
    in:  (DataFrame, shape=(1000, 50), dtype=float64, 400.0KB)
    out: (ndarray, shape=(1000, 128), dtype=float32, 500.0KB)

  _run_model: 44.50ms
    in:  (ndarray, shape=(1000, 128), dtype=float32, 500.0KB)
    out: (ndarray, shape=(1000, 10), dtype=float32, 39.1KB)

  postprocess: 0.03ms
    in:  (ndarray, shape=(1000, 10), dtype=float32, 39.1KB)
    out: (ndarray, shape=(1000,), dtype=int64, 7.8KB)
```

### Framework Collapsing

When `collapse_frameworks=True`, contiguous subtrees originating from
installed packages (detected via `site-packages` in the file path) are
collapsed into a single labeled node, regardless of the render depth.

```
predict: 48.12ms
  preprocess: 2.31ms
    validate_schema: 0.45ms
    normalize: 1.12ms
    [sklearn.preprocessing]: 0.74ms    ← collapsed
  _run_model: 44.50ms
    [torch.nn]: 44.48ms               ← collapsed
  postprocess: 0.03ms
    [numpy]: 0.02ms                    ← collapsed
```

This uses the `is_user_code` flag on each `SpanRecord`, determined by
checking whether the code object's `co_filename` is under a `site-packages`
directory. The heuristic is simple, correct for the vast majority of
environments, and overridable via a `user_modules` parameter.

### Export Formats

- **JSON**: Nested span tree with I/O summaries. Primary format for MLflow
  artifact logging and programmatic consumption.
- **Chrome Trace Event**: Duration events (`ph: "X"`) compatible with
  `chrome://tracing` and Perfetto for interactive flame charts.
- **Flat list**: Backward-compatible with the original module's `export()`
  format. Each span as a dict with `call_path`, `duration_ms`, etc.

---

## 5. Safety Guarantees

These are non-negotiable properties of the system.

### 5.1 Zero Side Effects When Disabled

When no `ProfileSession` is active:
- `@profile_span` decorators execute the wrapped function directly. The
  only overhead is `ContextVar.get(None)` (~30-50ns).
- `profile_block` context managers are no-ops.
- `sys.setprofile` is NOT installed. There is no callback, no event
  recording, no overhead beyond the decorator fast-path.

### 5.2 No Mutation of Model State

The profiler never:
- Modifies function arguments or return values
- Writes to model instance attributes
- Alters global or module-level state (except the scoped `ContextVar`)
- Holds references to user data objects (only `IOSummary` primitives)

### 5.3 Bounded Lifecycle

`sys.setprofile` is installed ONLY for the duration of a `with profiling()`
block or a single `autoprofile`-wrapped `predict()` call. It is removed in
a `finally` block, guaranteeing cleanup even on exception.

### 5.4 Exception Safety

If the profiler callback itself raises, it catches the exception, logs a
warning, and disables profiling for the remainder of the call. The model's
execution is never interrupted by a profiler failure.

If the model raises during profiling, the `finally` block in the context
manager ensures `sys.setprofile(None)` is called and the `ContextVar` is
reset.

### 5.5 Thread Isolation

`sys.setprofile` is per-thread. Each thread's profiling session is
independent. `ContextVar` provides per-context isolation, which is
correct for both threaded and async execution models.

Limitation: if `predict()` spawns internal worker threads, those are NOT
profiled by default. This is documented. Python 3.12+ offers
`threading.setprofile_all_threads()` as a future extension point.

### 5.6 Clean Patching/Unpatching

`autoprofile()` stores the original `PyFuncModel.predict` reference and
restores it exactly on `autoprofile(disable=True)`. Double-patching is
prevented by an idempotency guard.

---

## 6. Phased Approach

### v0 — Core Engine: Wall-Clock Profiling with Call Stack Capture

**Goal:** Prove the core mechanism works. A developer can wrap any
`predict()` call and get a structured, depth-controlled call tree with
wall-clock timing — with zero model changes.

**What gets built:**
- `sys.setprofile` callback with depth ceiling
- `ProfileSession` with flat event buffer and lazy tree reconstruction
- `SpanRecord` dataclass (timing fields only — no I/O yet)
- `profiling()` context manager
- `@profile_span` decorator and `profile_block` context manager
  (composable with auto-capture)
- `print_tree()` renderer with `depth` and `collapse_frameworks` options
- `to_json()` and `to_flat()` export
- `ContextVar`-based session lifecycle
- Overhead benchmarks proving the claims in §2.1

**What this validates:**
- The `sys.setprofile` + depth ceiling approach works and performs
  within budget
- The "capture once, render at any depth" property holds
- Manual decorators compose correctly with auto-captured frames
- Framework collapsing produces useful output on real models

**What this does NOT include:**
- No `autoprofile()` (you use `with profiling()` explicitly)
- No `IOSummary` / data capture
- No MLflow integration
- No packaging beyond a single installable source tree

### v1 — Productionize: Autoprofile, Packaging, Stability

**Goal:** Turn the core engine into a distributable, self-contained
Python package with the zero-effort `autoprofile()` entry point.

**What gets built:**
- `autoprofile()` with `PyFuncModel.predict` patching
- `autoprofile(disable=True)` clean unpatching
- `sample_rate` support for production use
- `last_profile()` retrieval
- `ProfileCollector` for batch aggregation with summary statistics
  (mean, p50, p95, p99 per span)
- Proper `pyproject.toml` with zero runtime dependencies
- `[mlflow]` optional extra for MLflow integration
- Comprehensive test suite: unit, integration, thread safety, noop
  overhead, end-to-end with MLflow model lifecycle
- API stability: public surface is documented and versioned

**Key design decisions at this phase:**
- Finalize the public API surface and commit to backward compatibility
- Decide on minimum supported Python version (3.9+ recommended)
- Decide on minimum supported MLflow version for the optional extra
- Establish the benchmark suite as a CI gate (regressions fail the build)

### v2 — I/O Tracking and UI Integration

**Goal:** Add data flow visibility and explore integration with MLflow's
existing tracing UI.

**What gets built:**
- `IOSummary` dataclass and `summarize()` function
- `summarize_locals()` for auto-captured frame arguments
- `capture_io` parameter on `profiling()` and `autoprofile()`
- I/O display in `print_tree()` (with `show_io` toggle)
- I/O fields in JSON export
- `register_summarizer()` extension point for custom types
- MLflow integration: `log_profile()` as artifact,
  `log_profile_metrics()` as run metrics,
  `log_profile_summary()` for aggregated stats
- Chrome Trace Event export (`to_chrome_trace()`) for Perfetto/
  chrome://tracing visualization

**UI integration investigation:**

MLflow's existing Tracing UI renders a span tree with timing, inputs,
and outputs — structurally identical to what we produce. The question
is whether we can write profiles in a format the Tracing UI can render
natively, or whether we need a custom visualization.

Options to evaluate:
1. **Emit OpenTelemetry-compatible spans** that the Tracing UI already
   understands. This would mean our profiles appear in the "Traces" tab
   alongside GenAI traces. Advantage: zero UI work. Disadvantage: we
   take on OTel as a dependency (even if only at export time), and
   the Tracing UI may not render our span types well (it's optimized
   for LLM/retriever/tool spans, not ML inference spans).
2. **Log profiles as HTML artifacts** using an embedded visualization
   (flame chart or tree view). The artifact viewer in the MLflow UI
   renders HTML natively. Advantage: full control over presentation,
   zero dependency on Tracing infrastructure. Disadvantage: not
   integrated into the experiment comparison workflow.
3. **Chrome Trace Event JSON as artifact** with a link to open in
   Perfetto. Lightweight, powerful visualization, no UI plugin needed.
4. **Custom MLflow UI plugin** (future). Full integration into the
   experiment view. Highest effort, highest payoff.

Recommendation: start with options 2 and 3 (HTML artifact + Chrome
Trace). Evaluate option 1 once the data model is stable. Reserve
option 4 for v3.

### v3 — Integration and Release

**Goal:** Pull everything together into a polished, tested, documented
package ready for public release and community adoption.

**What gets built:**
- Full documentation: README, API reference, usage guide, examples
- End-to-end examples: notebook workflow, FastAPI serving, batch
  inference pipeline
- Async export path: background thread + queue for non-blocking
  profile persistence in serving contexts
- Span processor hooks: `configure(span_processors=[fn])` for
  filtering, redacting, or enriching spans before export
- `c_call`/`c_return` configuration: opt-in capture of C-extension
  function calls (numpy ops, torch kernels)
- Evaluate and implement chosen UI integration path from v2
- Entry-point registration for MLflow plugin discovery
- CI/CD: automated testing, benchmark regression gates, PyPI publish
- Contribution guide for potential MLflow upstream contribution

**Post-v3 extensions (not scoped but designed for):**
- Custom collector plugins (CPU time, memory, GPU utilization)
- `threading.setprofile_all_threads()` support (Python 3.12+)
- Auto-discovery mode: walk model MRO and label methods automatically
- OpenTelemetry span export for Tracing UI integration
- Production alerting: shape drift detection, latency anomaly
  detection from profile stream

---

## 7. API Reference (by phase)

### v0 API

```python
from mlflow_profiler import profiling, profile_span, profile_block

# ── Context manager: wrap any code block ──────────────────
with profiling(depth=2) as session:
    result = model.predict(data)

session.print_tree()                    # Full captured depth
session.print_tree(depth=1)             # Aggregated view
session.print_tree(collapse_frameworks=True)

session.to_json()                       # JSON string
session.to_flat()                       # List of dicts
session.spans                           # list[SpanRecord]

# ── Decorator: label specific methods ─────────────────────
@profile_span("my_custom_label")
def preprocess(data):
    ...

# ── Block: label specific code regions ────────────────────
with profile_block("tensor_conversion"):
    tensor = to_tensor(features)
```

### v1 additions

```python
import mlflow_profiler

# ── Autoprofile: zero-effort ──────────────────────────────
mlflow_profiler.autoprofile(
    depth=2,
    sample_rate=1.0,
)

model.predict(data)
profile = mlflow_profiler.last_profile()
profile.print_tree()

mlflow_profiler.autoprofile(disable=True)

# ── Collector: batch aggregation ──────────────────────────
collector = mlflow_profiler.ProfileCollector()

for batch in batches:
    with collector.session(depth=2):
        model.predict(batch)

stats = collector.summary()
# {"predict": {"count": 100, "mean_ms": 48.1, "p50_ms": 47.3,
#              "p95_ms": 52.1, "p99_ms": 58.4},
#  "preprocess": {"count": 100, "mean_ms": 2.3, ...}, ...}
```

### v2 additions

```python
from mlflow_profiler import profiling

# ── I/O capture ───────────────────────────────────────────
with profiling(depth=2, capture_io=True) as session:
    result = model.predict(data)

session.print_tree(show_io=True)
session.print_tree(show_io=False)       # Timings only

# ── Custom summarizer ────────────────────────────────────
from mlflow_profiler import register_summarizer, IOSummary

register_summarizer(
    MyEmbedding,
    lambda obj: IOSummary(
        type_name="MyEmbedding",
        shape=(obj.num_vectors, obj.dim),
        dtype=str(obj.precision),
        size_bytes=obj.memory_footprint(),
    )
)

# ── MLflow integration ────────────────────────────────────
import mlflow
import mlflow_profiler

with mlflow.start_run():
    with profiling(depth=3, capture_io=True) as session:
        result = model.predict(data)

    mlflow_profiler.log_profile(session)          # JSON artifact
    mlflow_profiler.log_profile_metrics(session)  # Run metrics

# ── Chrome trace export ──────────────────────────────────
with open("profile.json", "w") as f:
    f.write(session.to_chrome_trace())
# Open in chrome://tracing or Perfetto
```

### v3 additions

```python
import mlflow_profiler

# ── Production autoprofile with async export ──────────────
mlflow_profiler.autoprofile(
    depth=2,
    capture_io=True,
    sample_rate=0.1,
    log_to_mlflow=True,     # Async background flush
)

# ── Span processors ──────────────────────────────────────
def redact_pii(span):
    if span.input_summary:
        for key in span.input_summary:
            if 'ssn' in key.lower():
                span.input_summary[key] = IOSummary(
                    type_name="REDACTED"
                )

mlflow_profiler.configure(span_processors=[redact_pii])

# ── Collector with MLflow logging ─────────────────────────
collector = mlflow_profiler.ProfileCollector()

for batch in batches:
    with collector.session(depth=2, capture_io=True):
        model.predict(batch)

with mlflow.start_run():
    mlflow_profiler.log_profile_summary(collector.summary())
```

---

## 8. Package Structure

```
mlflow-profiler/
├── pyproject.toml
├── README.md
├── src/
│   └── mlflow_profiler/
│       ├── __init__.py              # Public API surface
│       ├── _core.py                 # ProfileSession, SpanRecord, ContextVar
│       ├── _callback.py             # sys.setprofile callback
│       ├── _summarize.py            # IOSummary, summarize()        [v2]
│       ├── _decorators.py           # profile_span, profile_block
│       ├── _renderers.py            # print_tree, to_json, to_chrome_trace
│       ├── _autoprofile.py          # autoprofile() patching         [v1]
│       ├── _collector.py            # ProfileCollector               [v1]
│       └── mlflow_sink.py           # MLflow integration             [v2]
├── tests/
│   ├── test_core.py
│   ├── test_callback.py
│   ├── test_depth_aggregation.py
│   ├── test_decorators.py
│   ├── test_noop.py
│   ├── test_summarize.py                                           [v2]
│   ├── test_autoprofile.py                                         [v1]
│   ├── test_thread_safety.py
│   ├── test_collector.py                                           [v1]
│   └── test_mlflow_integration.py                                  [v2]
└── benchmarks/
    ├── bench_callback_overhead.py
    ├── bench_noop.py
    ├── bench_summarize.py                                          [v2]
    └── bench_end_to_end.py
```

### Dependencies

```toml
[project]
name = "mlflow-profiler"
requires-python = ">=3.9"
dependencies = []                   # Zero runtime dependencies

[project.optional-dependencies]
mlflow = ["mlflow>=2.10"]
dev = ["pytest", "pytest-benchmark", "mlflow>=2.10", "numpy", "pandas"]
```

---

## 9. Open Questions

These require discussion or investigation before implementation.

### `c_call` / `c_return` Events

`sys.setprofile` can fire on C-extension function calls (numpy ops, torch
kernels). These are where the actual compute happens but are far more
numerous than Python-level calls. A torch forward pass might generate
thousands of C-level events.

Options:
- Default on within the depth ceiling (consistent, but higher overhead)
- Default off, opt-in via `capture_c_calls=True` (safer default)
- Capture but collapse automatically into parent Python frame
  (best of both, more complex)

Recommendation: default off for v0/v1. Add as configurable in v2 with
the collapsing behavior. Benchmark the overhead difference to inform the
default.

### `frame.f_locals` Cost

Accessing `frame.f_locals` for I/O capture creates a dict copy
(~500ns-1μs per frame). At `depth=2` with ~10 captured frames, that's
~5-10μs additional overhead.

Options:
- Accept the cost (it's within the overhead budget at `depth=2`)
- Only capture I/O at the root span and `@profile_span`-decorated
  functions (which have direct `*args/**kwargs` access)
- Make I/O capture depth independently configurable:
  `capture_io_depth=1` to capture I/O only at the first level

Recommendation: accept the cost for v2. If benchmarks show it's
problematic at higher depths, add `capture_io_depth` as a separate knob.

### Multi-Threaded `predict()`

If a model's `predict()` spawns internal worker threads (e.g., parallel
feature extraction), those threads are not profiled by default because
`sys.setprofile` is per-thread.

Python 3.12+ offers `threading.setprofile_all_threads()`. For earlier
versions, we'd need to provide a `contextvars.copy_context()` helper
for manual propagation.

Recommendation: document the limitation. Do not attempt automatic
multi-thread profiling until v3.

### Sampling Strategy

`sample_rate=0.1` means "profile ~10% of calls." Should this be random
(each call has independent 10% probability) or periodic (every 10th call)?

Random avoids aliasing with periodic workload patterns but has variance.
Periodic is deterministic but can miss patterns.

Recommendation: random (`random.random() < sample_rate`), consistent
with MLflow Tracing's `MLFLOW_TRACE_SAMPLING_RATIO`.

### UI Integration Path

Discussed in v2 phase description. The main question is whether to target
the existing Tracing UI (requires OTel-compatible span format) or build
standalone visualization (HTML artifact + Chrome Trace). Needs
investigation into what the Tracing UI actually requires at the data layer
vs. what it could be made to accept.

---

## 10. Relationship to MLflow Tracing

`mlflow-profiler` is NOT a replacement for or competitor to MLflow Tracing.
They serve different use cases:

| | MLflow Tracing | mlflow-profiler |
|---|---|---|
| **Target** | GenAI call chains (LLM, retriever, tool) | Traditional ML inference (predict) |
| **Granularity** | Semantic spans (user-defined) | Call-stack frames (automatic) |
| **Overhead** | Higher (OTel + JSON serialization) | Lower (perf_counter_ns + list append) |
| **Dependencies** | OpenTelemetry, protobuf | None (stdlib only) |
| **Data captured** | Full message payloads, token counts | Lightweight I/O summaries |
| **Production mode** | Async logging, sampling | Async logging, sampling (same pattern) |
| **Integration** | 20+ GenAI library integrations | PythonModel / PyFuncModel focus |

The long-term aspiration is for `mlflow-profiler` to be a complementary
module in the MLflow ecosystem — the traditional-ML counterpart to GenAI
tracing. The API design intentionally mirrors MLflow Tracing conventions
(`@profile_span` ≈ `@mlflow.trace`, `profile_block` ≈ `mlflow.start_span`,
`autoprofile()` ≈ `mlflow.xyz.autolog()`) so that users moving between
GenAI and traditional ML workloads encounter familiar patterns.

If the project matures to the point of MLflow contribution, the natural
integration points are:
- Entry-point registration as an MLflow plugin
- Profile data logged as MLflow artifacts under runs
- Rendering in the MLflow UI (either via Tracing UI adaptation or a
  dedicated profiler view)
- `mlflow.profiler.autolog()` as a first-class autologging integration

---

## 11. Migration from Current Module

For reference, mapping from the existing profiling module to the proposed
API:

| Current | Proposed | Notes |
|---------|----------|-------|
| `threading.local()` | `ContextVar` | Async-safe |
| `uuid.uuid4()` per span | Integer index | ~100x cheaper |
| `start_profile_session()` | `with profiling():` | Scoped lifecycle |
| `get_profile(session_id)` | `session.to_flat()` | Method on session |
| `build_profile_tree()` | `session.to_tree()` | Depth-aware |
| `print_profile_tree()` | `session.print_tree()` | Unified, I/O-aware |
| `@profile_step("label")` | `@profile_span("label")` | Same syntax |
| Requires model refactoring | `sys.setprofile` auto-capture | Core improvement |
| No I/O visibility | `IOSummary` on every span | v2 |
| Global dict of sessions | ContextVar scoped session | No leak risk |
