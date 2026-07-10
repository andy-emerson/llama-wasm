# Feasibility + thin slice: one GGUF forward pass on WebGPU, no Emscripten

Date: 2026-07-09. Claim labels — **VERIFIED-LOCAL** (read in an installed
dependency or cloned source, cited), **WEB** (from a cited URL or a web-fetch
summary — weaker than reading the lines), **INFERRED** (reasoned from verified
facts), **UNCERTAIN** (needs an experiment or a deeper read). This document is
itself an early-rung artifact: it is grounded in a dependency spike (wllama
3.5.1, read locally) plus WEB reading of `ggml-webgpu` and the LlamaWeb paper,
with the architecture INFERRED. §8 lists what must be upgraded from WEB to
VERIFIED before the slice runs.

## TL;DR

A single-forward-pass proof of life is feasible, and its architecture is
largely **forced by decisions already made** (the durable statement is
`design/strategy.md` — **AOT the architecture, not the model**; this doc is the
feasibility grounding + go/no-go for its first slice):

1. **AOT the architecture into a static dispatch schedule — this is the root.**
   A transformer forward pass for a fixed architecture is a static compute
   graph; we compile it ahead of time into a fixed list of dispatch records and
   a JS driver plays it back. llama.cpp instead builds a graph at runtime and a
   C++ backend *walks* it — and that walker is the component that blocks on GPU
   sync (§3) and needs a C-API-over-JS bridge. **Removing the walker is what
   removes both problems.** [INFERRED; strategy.md §"why this is the whole
   ballgame"]
2. **Therefore JavaScript owns all asynchrony; wasm is pure compute.** This is
   the *consequence* of (1), not an independent choice: a dumb await-loop can
   drive the GPU only because the schedule is static. The driver `await`s
   `queue.onSubmittedWorkDone()` / `buffer.mapAsync()` natively; no blocking C++
   is ported, so the blocking wall never appears. [INFERRED from §3; WEB on the
   blocking pattern]
3. **WGSL compute shaders ported from `ggml-webgpu`'s templated quant kernels**,
   not written from scratch — we AOT the *schedule*, we *port* the kernels; that
   port is the real engineering center. [WEB]
4. **GGUF as the weight container**, parsed host-side, quantized blocks uploaded
   raw into the schedule's named buffer slots. Per-*architecture*, not
   per-model (vs WebLLM): a finetune/adapter is data, not a recompile. [INFERRED]
5. **Rust compute leftovers → `wasm32-unknown-unknown`**, minimal self-defined
   ABI, offset-based buffers, never assuming sole ownership of linear memory
   (Ruju's `rj_` discipline). [decision]

The slice **hand-writes one schedule** for one architecture+model — the AOT
architecture is present from day one; only the schedule *generator* is deferred
(strategy.md §"sequencing"), exactly as Ruju hand-builds one IR fixture before
building its producer.

**Thin slice (go/no-go):** the smallest real GGUF whose family we target, one
quantization format, one forward pass: GGUF parse → upload to WebGPU buffers →
hand-scheduled WGSL kernels → next-token logits. **Go** iff the argmax token
matches a llama.cpp CPU reference for a fixed prompt (top-k logits within a
stated tolerance), running with `crossOriginIsolated === false`, a
zero-Emscripten `wasm32-unknown-unknown` module, and no blocking wait anywhere.
Details in §7.

---

## 1. The problem restated (self-contained)

Two browser LLM stacks exist; both carry a tax (README expands this):

- **WebLLM (MLC/TVM)** [WEB]: only runs MLC-compiled models — a custom GGUF
  finetune cannot be dropped in. Disqualifying for this project's endgame (a
  custom Lua/LÖVE/LoveIDE model, see the LoveIDE agent plan).
- **wllama / llama.cpp-in-browser** [VERIFIED-LOCAL, `@wllama/wllama@3.5.1`]:
  runs any GGUF; API exposes `n_ctx`, `n_gpu_layers`, GBNF `grammar` sampling,
  `AbortSignal` cancel, OPFS caching, sharded GGUF, WebGPU auto-on since v3.1.
  Its cost: Emscripten-compiled WASM, and multi-thread wants COOP/COEP headers
  (README `esm/wllama.d.ts` / package README, read locally). Single-thread mode
  avoids the headers, but the Emscripten stack (loader, optional pthreads, the
  preloaded FS model) is the weight we want gone.

wllama remains the **working interim** behind LoveIDE's agent router and the
**performance baseline** to beat; llama-wasm is the from-scratch replacement.

## 2. Why not just port llama.cpp's C++ (the design we set aside)

The obvious "own build" is: compile llama.cpp to wasm without Emscripten, via
wasi-sdk (`wasm32-wasi`). Rejected for proof-of-life because **porting the C++
means porting the runtime graph-walker** — the very interpreter that blocks
in-band on GPU sync (§3) — so it inherits the blocking wall, and in the browser
also needs a WASI polyfill (a reintroduced system-interface layer). AOT-the-
architecture (strategy.md) sidesteps this at the root: there is no walker to
port because the graph is precompiled to a schedule. Porting the C++ stays a
*possible future* (it's what a `-wasi` name would imply); it is not what v0.1
pursues, and it's why the repo is `-wasm`, not `-wasi`. [INFERRED]

## 3. The blocking wall, and why removing the runtime walker dissolves it

`ggml`'s WebGPU backend (`ggml/src/ggml-webgpu/ggml-webgpu.cpp` in llama.cpp
mainline) is **~4,600 lines** of C++ against the standard `webgpu.h`/`webgpu_cpp.h`
C API — notably **not** Dawn-specific and with only a trivial
`#ifdef __EMSCRIPTEN__` include. [WEB — read via web-fetch summary, not yet
line-by-line; §8 task 1] It uses two families of WebGPU objects
(Instance/Adapter/Device/Queue/Buffer/ShaderModule/ComputePipeline/BindGroup/
CommandEncoder/QuerySet) and **blocks in-band**:

```cpp
instance.WaitAny( queue.OnSubmittedWorkDone(...), TIMEOUT );
instance.WaitAny( buffer.MapAsync(...), TIMEOUT );
```

Two layers make this hard to host in wasm. Both dissolve for the same reason —
**AOT precompiles the graph, so the runtime walker (this C++) is never in the
loop to bridge or to block** (strategy.md); "ownership inversion to JS" is how
that absence shows up in code:

- **Layer 1 — the C-API-over-JS bridge.** In a browser, WebGPU *is*
  `navigator.gpu` (JS). A wasm module reaches it only through imports someone
  writes: handle tables (C's opaque pointers ↔ JS objects), descriptor-struct
  marshalling, callback trampolines. Emdawnwebgpu is exactly this glue. [WEB]
  **Dissolved:** if the driver is JS, there is no C-to-JS bridge — JS calls
  `navigator.gpu` directly.
- **Layer 2 — blocking.** A browser thread cannot block: the promise behind
  `mapAsync` resolves only when the event loop turns, which a blocked wasm frame
  prevents ⇒ deadlock. Escapes are JSPI, Asyncify, or SAB+`Atomics.wait`
  (re-imports COOP/COEP). [INFERRED, standard event-loop semantics]
  **Dissolved:** `await queue.onSubmittedWorkDone()` / `await buffer.mapAsync()`
  is exactly how JS expresses these; the blocking idiom was a *native-runtime*
  habit. Remove the native runtime from the hot loop and there is nothing to
  block.

So the emscripten dependency is really a *"who runs the graph walker"*
dependency: host the walker and you owe it a GPU bridge and a suspension
mechanism. AOT deletes the walker — the schedule is data, the JS driver plays
it back, wasm is left with pure compute that needs neither bridge nor
suspension. [INFERRED — the load-bearing claim of the whole design; its proof
is §7.]

## 4. What we build against (state of the pieces)

- **WebGPU compute in browsers** [WEB]: WGSL compute shaders + storage buffers;
  Chrome ships it; Firefox/Safari coverage is partial and must be checked at
  slice time (§8 task 4). Node has WebGPU implementations usable for headless
  Tested/Oracle rungs; the Browser rung stays owed.
- **`ggml-webgpu` quant kernels** [WEB]: the WGSL/templated kernels for GGUF
  quant formats (Q4_K, Q8_0, …) are the porting source. This is where the
  LlamaWeb paper's IP lives; porting them (not reinventing) is the center of
  gravity. Must be read line-by-line before porting (§8 task 1).
- **GGUF format** [WEB]: documented container — header, metadata KV, tensor
  table, quantized blocks. A host-side parser reads it; weights upload raw.
  ArrayBuffer's 2 GB limit and WebGPU's `maxBufferSize` /
  `maxStorageBufferBindingSize` force tensor chunking (§6, §8 task 4).
- **llama.cpp CPU** [decision]: the numerical oracle. `llama-cli`/`llama-eval`
  style output for a fixed (model, prompt), pinned by commit, is the reference
  every kernel and the final logits are graded against.

## 5. Value representation and ABI

- **The heavy path never enters wasm.** Weights live in GPU buffers; activations
  live in GPU buffers; the driver orchestrates in JS. Wasm holds only the
  small pure-compute leftovers.
- **wasm compute leftovers** (tokenizer, sampler, dequant helpers for anything
  not done on-GPU): `wasm32-unknown-unknown`, `panic=abort`, near-zero imports,
  a minimal `lw_`-prefixed export surface, all buffers passed as
  `(ptr:u32, len:u32)` offsets into linear memory the module does not assume it
  owns (Ruju's composability rule). [decision, mirrors Ruju's reasoning; not a
  port of it]
- **No WASM-GC, no WASI.** Nothing here needs GC or OS services. [INFERRED]

## 6. Constraints that shape the slice

- **Buffer limits** [WEB, verify §8.4]: large weight tensors exceed a single
  WebGPU buffer's `maxStorageBufferBindingSize` on many GPUs; the schedule must
  chunk matmuls. Even the slice must confront this for a real model's embedding
  / output projection.
- **Per-dispatch overhead** [INFERRED]: a forward pass is many small kernel
  dispatches per token; a JS driver adds per-dispatch cost. Fine at proof-of-
  life (correctness, not speed), measured (§7) but not gated.
- **Numerical order** [INFERRED, the deepest risk — §8 risk 1]: dequant +
  matmul accumulation order in a ported WGSL kernel can differ from llama.cpp's,
  drifting logits. Use fp32 accumulation, one well-understood format first, and
  a stated tolerance.

## 7. The thin-slice experiment (go/no-go)

**Goal:** prove the whole chain — GGUF → WebGPU → logits — end to end, on the
smallest honest slice, before any of the XL (schedule compiler, KV cache, quant
zoo) is committed. Mirrors Ruju's thin-slice discipline; the target is *our*
oracle (llama.cpp numbers), not a C port.

**Fixture:** the smallest real GGUF of the target family (e.g. a ~0.5B Qwen2.5
class model — small enough to iterate, real enough that llama.cpp CPU gives a
meaningful reference), **one quant format** — start **Q8_0** (simplest dequant,
least numerical ambiguity), then **Q4_K** (the format real models ship). A
fixed short prompt.

**Steps** (each independently verifiable, each graded on the ladder):

1. **GGUF parse + one dequant.** Host-side parse of header/metadata/tensor
   table; dequantize one weight tensor. *Oracle:* dequantized values equal
   llama.cpp's for that tensor within tolerance.
2. **One matmul kernel.** Port one `ggml-webgpu` quant-matmul WGSL kernel;
   dispatch from JS; read back via `await buffer.mapAsync()` — the exact op the
   C++ blocks on, done async. *Oracle:* result equals a llama.cpp CPU matmul of
   the same operands.
3. **One full forward pass.** Hand-schedule in JS: embed → n_layer × (RMSNorm →
   attention w/ RoPE → RMSNorm → SwiGLU FFN) → final norm → output projection →
   logits, over WGSL kernels. *Oracle:* argmax next-token equals llama.cpp CPU
   for the fixed prompt; top-k logits within tolerance.
4. **The `wasm32-unknown-unknown` proof.** Tokenizer *or* sampler in Rust →
   `wasm32-unknown-unknown`, loaded as a plain ES module, called synchronously
   between GPU steps. *Oracle:* tokenizer output equals llama.cpp's token ids
   for the prompt / sampler picks the same token as a reference decode.

**Go/no-go thresholds:**

| Measurement | Threshold | Rationale |
| - | - | - |
| Correctness | argmax next-token **exactly** equals llama.cpp CPU on the fixed prompt; top-5 logits within a **stated relative tolerance** (propose 1e-2, set from the fp32-vs-fp16 accumulation gap observed in step 2) | non-negotiable; the whole point is a *faithful* forward pass |
| No isolation | runs with `crossOriginIsolated === false` | the reason the project exists |
| No Emscripten | the wasm module built with `wasm32-unknown-unknown`, zero Emscripten in the toolchain; loads as an ES module — no `Module` glue, no service worker | the escape, made concrete |
| No blocking | the driver never calls a blocking wait; every GPU sync is `await` | structural — confirmed by inspection, not measured |
| Speed sanity | decode at interactive rate; soft-compare to wllama on the same model+GPU | **not a gate at PoL**; recorded to size the later schedule-compiler work |

**Explicit non-goals of the slice** (deferred): the schedule *compiler* (this
slice hand-writes one schedule); KV cache beyond one short pass; the full quant
zoo (two formats only); LoRA/adapters; multi-architecture; performance tuning;
single-file packaging; the tokenizer's full vocab/merges if a smaller fixture
proves the mechanism.

## 8. Pre-slice verification (upgrade WEB → VERIFIED before running)

The honest gate before writing kernel code:

1. **Read `ggml-webgpu.cpp` line-by-line** (clone llama.cpp at a pinned commit):
   confirm the kernel set, the quant-matmul WGSL, the exact wait/readback shape,
   and that the compute kernels carry no hidden Emscripten/host coupling.
   Upgrades §3–§4 from WEB to VERIFIED-LOCAL.
2. **Choose the model + capture the oracle:** pick the fixture GGUF; run
   llama.cpp CPU at a pinned commit; check in the reference logits/tokens/tensor
   fixtures with the exact command and commit that produced them.
3. **Confirm the quant math:** read the Q8_0 and Q4_K block layouts and dequant
   formulas in the pin; they drive both the host parser and the WGSL.
4. **Probe WebGPU on target GPUs:** `maxStorageBufferBindingSize`,
   `maxBufferSize`, fp16 support, and the Firefox/Safari coverage question —
   these size the chunking strategy and the Browser-verified reachability.

## 9. Risk register

| Risk | Severity | Mitigation |
| - | - | - |
| **Numerical fidelity** — ported WGSL vs llama.cpp accumulation order (§6) | High (correctness) | per-kernel oracle checks before composing; fp32 accumulation; Q8_0 first; tolerance stated up front |
| Kernel-porting is the real engineering mass (§4) | High (effort) | port, don't reinvent; one format, one matmul first; the slice deliberately touches the minimum |
| WebGPU browser coverage (Chrome vs FF/Safari) (§6) | Medium | Chrome-first for Browser-verified; Node WebGPU for headless rungs; record coverage, don't gate PoL on FF/Safari |
| Buffer-size limits force chunking (§6) | Medium | confront in the slice's output projection; design the schedule chunk-aware from step 2 |
| Per-dispatch JS overhead at many kernels/token (§6) | Medium, later | measure at PoL (non-gating); it sizes the schedule-compiler / batching work |
| `wasm32-unknown-unknown` ↔ JS ABI friction | Low | minimal `lw_` surface, offset buffers; the Rust-wasm web target is mature |
| GGUF 2 GB ArrayBuffer limit for big models | Low, later | sharded/streamed load; irrelevant at the small-fixture PoL |

## 10. Verdict

**Feasible, architecture essentially determined by prior decisions.** The one
load-bearing bet — that inverting async ownership to JS makes both the
Emscripten bridge and the blocking wall *not exist* rather than need solving —
is exactly what the thin slice proves or refutes, cheaply, before any XL is
committed. The deepest risk is numerical (ported-kernel fidelity vs the
oracle), not architectural, and it is bounded by grading every kernel against
llama.cpp before composing and starting from the simplest quant format.
Recommended next increment: **§8 (pre-slice verification)** — clone and read
`ggml-webgpu` at a pin, capture the oracle, then step 1 of §7.

---

### Deviation table

| Planned | Delivered | Rung | Severity | Recorded |
| - | - | - | - | - |
| A claim-labelled feasibility + thin-slice plan | This document | **Stated** (plan), grounded by a VERIFIED-LOCAL wllama spike + WEB reading of `ggml-webgpu`/LlamaWeb | — | here |
| Feasibility "verified" | **Not claimed** — `ggml-webgpu` is WEB (web-fetch summary), not read line-by-line; no oracle captured yet; no browser/GPU access in the authoring sandbox | below Oracle-verified by construction | Medium (must clear §8 before kernel code) | §8 |
