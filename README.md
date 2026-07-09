# llama-wasm

Run a GGUF language model in the browser on **WebGPU**, without Emscripten,
without cross-origin isolation, and without an in-wasm blocking runtime.

llama-wasm is **early-stage — a plan, not yet a runtime.** This repository
currently holds the design: the feasibility research and the go/no-go
proof-of-life experiment. No code has been written yet.

## The problem

Running an LLM in the browser today means one of two stacks, and both carry a
tax we want to shed:

- **WebLLM (MLC/TVM)** runs only models someone has compiled through the MLC
  pipeline — a custom finetune can't just be dropped in.
- **wllama / llama.cpp-in-the-browser** runs any GGUF, but reaches the browser
  through Emscripten-compiled WebAssembly, and its multi-thread path wants the
  `Cross-Origin-Embedder-Policy` / `Cross-Origin-Opener-Policy` headers
  (cross-origin isolation). Emscripten's loader, its optional pthreads/COOP-COEP
  machinery, and its preloaded filesystem are exactly the weight a from-scratch
  browser runtime shouldn't have to carry.

There's a deeper wall behind the second option. llama.cpp's WebGPU backend is
C++ that **blocks in-band** on GPU work (`instance.WaitAny(...)` around queue
submission and buffer readback). A browser's main thread cannot block — the
promise a GPU operation resolves only fires when the event loop turns, which a
blocked wasm call prevents. Getting blocking C++ into a browser needs JSPI,
Asyncify, or SharedArrayBuffer-plus-`Atomics.wait` (which re-imports the very
COOP/COEP headers we're escaping).

## The approach

**AOT the architecture, not the model.** A transformer forward pass for a fixed
architecture is a *static* compute graph — the same ordered kernel dispatches
every token. Compile it ahead of time into a fixed **dispatch schedule** and a
plain JS driver plays it back. llama.cpp instead builds a graph at runtime and a
C++ backend *walks* it — and that walker is the component that blocks on GPU
sync and needs a C-API-over-JS bridge. **Delete the walker and both problems
never appear.** So the blocking problem doesn't get solved — it stops existing.
The full statement is [`design/strategy.md`](design/strategy.md); the rest
follows from it:

- **JavaScript owns all asynchrony; wasm is pure compute** — the *consequence*
  of AOT, not an independent choice. A **plain-JS driver** issues the schedule's
  WebGPU dispatches and `await`s every GPU sync (`queue.onSubmittedWorkDone()`,
  `buffer.mapAsync()`) natively — the idiom JavaScript is built for. A dumb
  await-loop can be in charge only because the schedule is static.
- The heavy matmuls run as **WGSL compute shaders**, ported from llama.cpp's
  `ggml-webgpu` templated quantization kernels (the real engineering center),
  not reinvented.
- **GGUF is the weight container**, parsed host-side; quantized weights are
  uploaded to the schedule's buffer slots as raw data. Because the schedule is
  compiled per *architecture*, not per *model* (WebLLM's per-model compile is
  its cage), a new finetune or LoRA adapter is *data*, never a recompile.
- The compute leftovers (tokenizer, sampler, dequant helpers) are small, pure
  Rust compiled to **`wasm32-unknown-unknown`** — no system interface to shim,
  a minimal self-defined ABI, loaded as a plain ES module. (WASI is the right
  target for porting a C program that needs libc; it is the wrong target here,
  because in the browser it means shipping a WASI polyfill — reintroducing the
  system-interface layer this project exists to delete. Hence `-wasm`, not
  `-wasi`.)

The v0.1 goal is **proof of life**: one forward pass of one small GGUF model,
producing next-token logits that match a llama.cpp CPU reference, in a browser,
with no isolation and no blocking. Everything after that — the schedule
compiler, the KV cache, the quantization-format zoo, adapters — is deferred
until that chain is proven. See
[`design/research/research-forward-pass.md`](design/research/research-forward-pass.md).

## On Ruju as inspiration

The methodology here is adapted from the sibling
[Ruju](https://github.com/andy-emerson/Ruju) project (Julia-in-Rust-on-wasm):
the claim ladder, the thin-slice-before-XL discipline, the composable ABI that
never assumes sole ownership of linear memory. But llama-wasm is **not** a port
the way Ruju is. Ruju faithfully reimplements a C runtime, verified structure-
by-structure against pinned C. llama-wasm is a *fresh architecture* — the JS
driver and the static dispatch schedule are ours, not anyone's port. Our only
external reference is a **numerical oracle** (does our forward pass match
llama.cpp's output?), plus `ggml-webgpu`'s kernels as a porting source for the
WGSL specifically. Ruju is a solution-shaped inspiration, not an oracle to
imitate.

## Layout

| Path | What it is |
| - | - |
| `design/strategy.md` | the durable architectural thesis: AOT the architecture, not the model |
| `design/methodology.md` | how work moves from written to pushed — the claim ladder and the increment loop |
| `design/research/` | claim-labelled feasibility research and go/no-go experiment plans |

## License

MIT ([LICENSE](LICENSE)).
