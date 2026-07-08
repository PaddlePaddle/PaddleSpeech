# Notes for agents/contributors running `paddlespeech tts` (verified on Tesla T4, Turing sm_75)

This file documents things confirmed by actually running the documented `paddlespeech tts` CLI
end-to-end on a real NVIDIA Tesla T4 (16GB, driver 550.163.01 / CUDA 12.4, paddlepaddle-gpu 3.3.1
cu126 build, Python 3.10.12). Nothing here changes any model code or default behavior — this is
purely install/config guidance for the next person (human or agent) who runs this repo.

## 1. Install `paddlepaddle-gpu`, not `paddlepaddle` — the README's own example is CPU-only

The Installation section's copy-pasteable example is:

```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

This installs the **CPU-only** wheel (the package is literally named `paddlepaddle`, not
`paddlepaddle-gpu`). If you run this on a GPU machine, nothing errors — `paddlespeech tts` still
runs to completion — but every subsequent inference silently falls back to CPU, because
`paddlespeech`'s default `--device` is `paddle.get_device()`, which correctly reports `cpu` when
paddle wasn't compiled with CUDA support. There is no warning that the GPU is unused.

Verified:
```
$ pip install paddlepaddle   # exact package name from the README example
$ python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.get_device())"
False cpu
```

Measured cost of following the README literally on a GPU box (`fastspeech2_csmsc` + `hifigan_csmsc`,
same sentence, same weights, inference-only time excluding one-time tokenizer/frontend load):
- CPU (`--device cpu`, i.e. what you get if you followed the README verbatim): **6.846s, 6.850s** (2 runs)
- GPU (`--device gpu:0`): **1.054s, 1.050s** (2 runs)

That's a ~6.5x slowdown with zero indication anything is wrong.

**Recommendation:** for a CUDA-capable machine, install `paddlepaddle-gpu` instead, matched to your
driver's CUDA support. On this box (driver 550.163.01, native max CUDA 12.4), the wheel built for
CUDA 12.6 worked fine via CUDA's minor-version compatibility — confirmed with `paddle.utils.run_check()`:

```
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
```
$ python -c "import paddle; paddle.utils.run_check()"
...
W ... Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 12.4, Runtime API Version: 12.6
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 GPU.
PaddlePaddle is installed successfully!
```
(cu123, matching the driver's native version more closely, is also available at
`https://www.paddlepaddle.org.cn/packages/stable/cu123/` but only ships paddlepaddle-gpu 3.1.0 for
Python 3.10 at time of writing — 3.3.1/cu126 was used for this verification instead.)

## 2. `pip install .` can fail with `ImportError: cannot import name 'download' from 'aistudio_sdk.hub'`

Reproduced on a fresh venv, `pip install .` from repo root (source-compilation install path from the
README):

```
ImportError: cannot import name 'download' from 'aistudio_sdk.hub'
```

Cause: `paddlenlp==2.8.1` (pulled in transitively by this repo's `setup.py`) declares
`aistudio-sdk >=0.1.3` with no upper bound. `pip` resolves this to the newest release
(`aistudio-sdk==0.3.8` at time of writing), but the `download` function was removed/renamed out of
`aistudio_sdk/hub.py` somewhere between 0.2.6 and 0.3.0 (confirmed by downloading both wheels from
PyPI and diffing `hub.py` — 0.2.6 has `def download(**kwargs)` at line 655, 0.3.0 and 0.3.5 do not).
`paddlenlp/transformers/aistudio_utils.py` still does `from aistudio_sdk.hub import download`, so
any newer `aistudio-sdk` breaks import of `paddlespeech` entirely (not just TTS — `paddlenlp` is
imported early in the `t2s` frontend chain for the Chinese G2P/BERT-based polyphone disambiguation).

**Fix that worked:**
```
pip install "aistudio-sdk==0.2.6"
```
applied *after* `pip install .` (pip's resolver will otherwise upgrade it back if pinned before).

## 3. T4 (Turing, sm_75) acceleration checklist — measured, not assumed

Scope: default `paddlespeech tts --input "..." --output output.wav"` CLI path only, i.e.
`fastspeech2_csmsc` (acoustic model) + `hifigan_csmsc` (vocoder), both dynamic-graph (`paddle.nn.Layer`)
inference via `TTSExecutor`. `paddlespeech/t2s/exps/inference.py` / `PTQ_static.py` (Paddle-Inference
predictor export path with fp16/bf16/int8 options) is a **separate, undocumented-in-README** deployment
route and was not the target of this check — see note in §3d.

**a) dtype:** both models load as `float32` by default (confirmed:
`next(iter(am_inference.parameters())).dtype == paddle.float32`, same for the vocoder — no
`torch_dtype`-equivalent config leak, this is a clean default). Forcing fp16 via
`paddle.amp.auto_cast(level="O1")` around the same `.infer()` call is **slower**, not faster:
fp32 baseline 1.048s/1.060s vs O1 autocast 1.326s/1.248s (2 runs each, instrumented,
`paddle.device.synchronize()` around the call). Forcing pure fp16 via
`paddle.amp.decorate(models=..., level="O2")` is also slower (1.263s/1.263s) **and breaks the
standard output-writing step**: the synthesized waveform tensor comes back as `float16`, and
`TTSExecutor.postprocess()` (via `soundfile`) raises
`ValueError: dtype must be one of ['float32', 'float64', 'int16', 'int32'] and not 'float16'`
(reproduced twice). **Recommendation: do not force fp16/AMP for this CLI path** — the implicit fp32
default is already the best of the three options measured, and O2 additionally breaks WAV writing
outright.

**b) attention backend:** `paddlespeech/t2s/modules/transformer/attention.py`'s `MultiHeadedAttention`
is a hand-written matmul+softmax implementation; it does not call Paddle's native
`paddle.nn.functional.scaled_dot_product_attention` / `flash_attention` APIs anywhere (confirmed via
`grep -rn "scaled_dot_product_attention\|flash_attn\|flash_attention" paddlespeech/t2s/` → 0 matches;
the one repo-wide match is in the unrelated ASR `s2t/models/wavlm` module). Side-check on this same
T4 box: Paddle's own flash-attention kernel explicitly rejects Turing —
`paddle.nn.functional.flash_attention.flash_attention(q, k, v)` on fp16 tensors raises
`OSError: ... is_sm8x || is_sm90_or_larger check failed ...`, and forcing the backend via
`sdpa_kernel([SDPBackend.FLASH_ATTENTION])` raises
`RuntimeError: No available backend for scaled_dot_product_attention was found` — `EFFICIENT_ATTENTION`
and `MATH` backends work fine. This mirrors PyTorch's `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`
behavior on sm75 GPUs. Not actionable for this repo without a model-code change (out of scope here) —
noted for context only.

**c) int8 / quantization:** `paddleslim` is installed as a transitive dependency, but its only usage
in this repo is `paddlespeech/t2s/exps/PTQ_static.py` and `PTQ_dynamic.py` — the offline
Paddle-Inference export/quantization tooling, not wired into `TTSExecutor`'s dynamic-graph CLI path.
`grep -rn "bitsandbytes" paddlespeech/` → 0 matches anywhere in the repo.

**d) CUDA Graphs:** Paddle has a native `paddle.device.cuda.graphs.CUDAGraph` API, but
`grep -rn "cuda_graph\|CUDAGraph\|cudaGraph" paddlespeech/` → 0 matches; not used anywhere in this
repo. Not attempted independently here — `TTSExecutor`'s input is variable-length text, which
conflicts with CUDA graphs' static-shape/address requirement, and working around that would mean
changing how the model is invoked rather than documenting existing behavior (out of this track's
doc-only scope).

`torch.compile` / any Paddle JIT-compile equivalent was intentionally not evaluated (out of scope for
this pass).

## 4. Determinism

Unlike some autoregressive TTS models, `fastspeech2_csmsc` + `hifigan_csmsc` are fully
non-autoregressive — repeated runs of the exact same input produce a bit-identical output length
(85500 samples / 24kHz = 3.5625s, verified across 3 separate documented-command runs). No seed
management is needed for reproducibility with this default pipeline.

## 5. Measured reference numbers (this box only, not a general benchmark claim)

- GPU: Tesla T4 16GB, driver 550.163.01, CUDA 12.4 (driver) / paddlepaddle-gpu 3.3.1 built for cu126
- Wall time, documented command (`paddlespeech tts --input ... --output output.wav`, cache warm,
  includes CLI startup/tokenizer load): 23.334s, 32.069s (2 runs)
- Peak VRAM (`nvidia-smi`, whole-process, background-sampled at 1Hz): 1707 MiB (both runs)
- Pure model-inference time (instrumented, excludes one-time tokenizer/frontend load): 1.048s, 1.060s
