# Strategy: AOT the architecture, not the model

Where llama-wasm is going, and the one idea the whole design turns on. The
feasibility work and the proof-of-life experiment live in `research/`; this is
the durable architectural thesis they serve.

## The core idea

A transformer forward pass, for a **fixed architecture**, is a *static* compute
graph: the same ordered sequence of kernel dispatches, every token, forever.
Nothing about it needs to be decided at runtime. llama-wasm compiles that graph
**ahead of time** into a fixed **dispatch schedule** and hands it to a trivial
JavaScript playback loop. There is no runtime graph builder and no runtime
interpreter walking the graph — which is precisely the component that, in
llama.cpp, is blocking C++.

## Why this is the whole ballgame

- llama.cpp builds a `ggml_cgraph` at runtime and a backend **walks** it,
  dispatching ops and blocking in-band on GPU sync (`instance.WaitAny(...)`).
  That walker is the native C++ that (a) can't be reached from wasm without a
  C-API-over-JS bridge and (b) can't block on a browser thread. Hosting it is
  exactly what drags in Emscripten, JSPI, or SharedArrayBuffer.
- **Remove the walker and there is nothing to port and nothing to block.** The
  schedule is precomputed data; a JS driver plays it back, `await`ing each GPU
  sync natively. So **AOT is the cause; "JavaScript owns async, wasm is pure
  compute" is its consequence.** A dumb await-loop can only be in charge because
  the graph was made static ahead of time — if the next op had to be decided at
  runtime, you'd need the interpreter back, and the interpreter is the blocking
  C++.

This is the specific thing that distinguishes llama-wasm from "just use wllama"
or "just port llama.cpp": both of those keep the runtime walker. We delete it.

## Two precisions that make "AOT" mean the right thing here

1. **Per-architecture, not per-model — the relaxation vs WebLLM.** WebLLM/TVM
   AOT-compiles per *model*: its strength and its cage — you can run only what
   someone compiled. But the static graph is a property of the *architecture
   family* (Qwen2.5, Llama, …), not of the weights. Compile **one schedule per
   architecture**; weights are data bound into named buffer slots at load. A new
   finetune or LoRA adapter is *data*, never a recompile. This is what keeps
   "run my own GGUF" alive while still getting AOT's no-interpreter benefit —
   the exact property the LoveIDE custom-model endgame needs.
2. **Kernels ported, not generated — the divergence vs TVM.** TVM *generates*
   kernels. We **port** `ggml-webgpu`'s WGSL quant kernels (its real IP) and AOT
   only the **schedule** — the ordered dispatch graph that binds those kernels
   to buffers for a given architecture. So "AOT" in llama-wasm compiles the
   *schedule*, not the kernels.

## What actually gets compiled

From an architecture's hyperparameters (n_layers, n_heads, d_model, ffn dims,
quant format, RoPE params, …) → a fixed list of **dispatch records**:
`(kernel id, bind-group layout, workgroup dims, buffer bindings, chunking for
buffer-size limits)`. Weights fill the named buffer slots at load time; the KV
cache and sampling hang off the ends. The JS driver executes the list per
token — bind, dispatch, `await` sync, repeat — and nothing more.

## Sequencing: hand-author the schedule before you generate it

The architecture is AOT **from day one**. What the proof-of-life defers is the
*generator*, not the AOT-ness: v0.1 **hand-writes one schedule** for one
architecture + model, exactly as Ruju hand-constructs one IR fixture before
building its producer. Proving that a hand-authored static schedule + ported
kernels reproduce llama.cpp's logits is what earns the right to build the
**schedule compiler** (architecture hyperparams → schedule) that comes after.
The first compiled artifact is written by hand; the compiler that emits the
rest is the subsequent XL.

See `research/research-forward-pass.md` for the feasibility grounding and the
go/no-go experiment that proves this spine.
