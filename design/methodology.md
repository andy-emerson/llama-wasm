# Methodology

How work moves from written to pushed in a project that runs on human–agent
trust and demands numerical precision. Adapted from the sibling
[Ruju](https://github.com/andy-emerson/Ruju) project — its claim ladder and
increment loop — but **not its framing**. Ruju is a faithful reimplementation
of a C runtime, graded against pinned C structure-by-structure. llama-wasm is a
*fresh architecture*: the JS driver and the static dispatch schedule are ours,
not a port of anyone's code. So our top rung is not "matches the C" — it is
"matches llama.cpp's numbers, in a real browser."

The founding observation, inherited unchanged: **the failure mode is never
fabrication — it is a claim quietly sitting one rung above its evidence.**
"The kernel runs" drifts into "the kernel is correct." "The logits look
plausible" drifts into "the forward pass matches." The ladder exists to make
that drift visible.

## The claim ladder

Every statement about correctness carries a grade. A claim may never be
reported at a rung above its evidence.

| Grade | Meaning | Evidence |
| - | - | - |
| **Stated** | asserted; no evidence yet | — |
| **Tested** | passes our own tests | native `cargo test`, the wasm harness, WGSL kernel unit checks — all green, zero warnings |
| **Oracle-verified** | output matches llama.cpp's own numbers | a dequantized tensor, a matmul result, or next-token logits equal a llama.cpp CPU reference for the same (model, input), within a stated tolerance, cited by the reference-capture script |
| **Browser-verified** | runs in a real browser on real WebGPU | loaded in a browser with a GPU, produced the oracle-matching result there, with `crossOriginIsolated === false` |

Our own tests can encode the same misunderstanding as the code — **Tested is
where verification starts, not where it ends.** A WGSL kernel that passes a
hand-written unit test can still disagree with llama.cpp on accumulation order;
only the oracle catches that. And a kernel that matches the oracle under Node's
WebGPU can still fail on a real GPU's precision or buffer limits — only the
browser catches that.

**Two rungs are, by their nature, owed rather than claimable in a headless
sandbox** (the same honesty LoveIDE's ladder demands of love.js/WebGPU): a dev
machine without a GPU can reach *Tested* and, via a CPU WebGPU implementation,
approach *Oracle-verified*, but **Browser-verified is only reachable by a human
loading it on a GPU.** Say so plainly; never dress an owed rung as reached.

The ladder applies **symmetrically to claims about the references**. "llama.cpp
does X" / "`ggml-webgpu` computes Y this way" / "WebGPU guarantees Z" carries a
citation (a file:line into a cloned llama.cpp, a spec section) or it is merely
Stated — never ground truth. Reconstructed-from-memory characterizations of a
dependency are the classic over-claim; this project reads the source and cites
it.

## The increment loop

1. **Pick from the frontier.** The human selects the increment and holds design
   authority over scope. The current frontier is the proof-of-life forward pass
   (`design/research/research-forward-pass.md`).
2. **Establish the oracle first.** Before writing a kernel or a pass, capture
   the llama.cpp CPU reference it must match — the dequantized tensor, the
   matmul output, the logits — as a checked-in fixture with the exact (model,
   input, llama.cpp commit) that produced it. The reference is captured at the
   start so the post-write check is confirmation, not discovery.
3. **Write the code and its test together.** The test must be able to fail:
   compare against the oracle fixture, not against the code's own output. A fix
   ships in the same commit as the check that would have caught the defect.
4. **Verify it works, in order.** `cargo test` (native), a zero-warnings build,
   the wasm build (`wasm32-unknown-unknown`), the Node harness, the oracle
   comparison, a whitespace check on changed files. Any failure stops the loop.
5. **Grade honestly.** Record the rung each claim actually reached. Kernel
   matches the oracle under Node → *Oracle-verified*, browser run *owed*. No
   GPU in the sandbox → say the browser rung is unmet. **The push gate is
   "honestly graded," not "fully verified."**
6. **Update the documents in the same commit.** Research/status docs carry the
   new rung and any divergence; a doc that lags the code is the first step of
   the next over-claim.
7. **Commit and push** per the repo's conventions.
8. **Report with a deviation table.** Every increment report ends with a table:
   one row per departure from the slice's plan and per claim sitting below
   Oracle-verified — planned vs delivered, the claim's rung, a severity, where
   it's recorded — sorted by severity. **An empty table is itself a claim
   ("no deviations") that the next audit will check.**

## Divergences and pins

- **Divergences** from a reference (a WGSL kernel that restructures
  `ggml-webgpu`'s dispatch, a value layout that differs) are never silent: the
  status row says *Divergence* with a dated note on exactly what differs and
  why, and — where behavior could drift — an oracle case that keeps testing it.
- **Pin discipline:** llama.cpp (the kernel-porting source and the numerical
  oracle) and any WebGPU/GGUF spec claims are pinned to a recorded commit /
  version. Ported kernels cite the pin; advancing the pin is its own increment —
  re-vendor, re-capture affected oracle fixtures, re-audit ported kernels.

## Audits

A quick orientation pass at the start of each session — the documents match the
code, the frontier matches reality — before new work is picked up; a deeper
pass when a claim is promoted to Oracle- or Browser-verified (the promotion
*is* an audit finding, with its evidence), when the pin advances, or at the
human's discretion. The goal is not a clean audit but an honest ledger.
