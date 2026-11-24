# Qwen3.8-Flash-Next-FP4-Local on Windows

Scripts for serving [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
with this vLLM fork on one RTX PRO 6000 Blackwell (sm_120, 96 GB), using the
model card's PRO 6000 configuration: MTP speculative decoding (3 tokens), b12x
linear and MoE kernels, FULL_AND_PIECEWISE CUDA graphs, 262K context, FP8 KV
cache, async scheduling, prefix caching, and the reasoning and tool-call parsers.

Tested with Windows 11, driver 621.24, CUDA 13.4, VS 2026 (MSVC 14.52),
Python 3.14, PyTorch 2.15 and vLLM 0.34.0 from the `vllm-for-windows` branch.

Measured on that machine, one request at a time: 150–225 tokens/s decode,
about 2.7 tokens accepted per MTP step, 539K tokens of KV cache, about
7 minutes from launch to ready.

## Files

| File | Purpose |
|---|---|
| `buildenv.cmd` | Build environment for the wheel: VS 2026 compiler, CUDA 13.4, sm_120 only. Runs its arguments from the repo root. |
| `runenv.cmd` | Runtime environment: MSVC (for FlashInfer's JIT), `CUDA_HOME` and the repo's `.venv`. Runs its arguments. |
| `serve.cmd` | The server: model card settings adapted to Windows (see below). `GPU_UTIL`, `SERVE_HOST` and `SERVE_PORT` override the defaults (0.88, 127.0.0.1, 8000); extra arguments go to `vllm serve`. |
| `startup.cmd` | Logon variant: waits for `SERVE_HOST` to come up, serves extra model names from `SERVED_NAMES`, logs to `startup.log`. Set `VLLM_API_KEY` to require a key. |
| `patch_fcntl.py`, `winflock.py` | Patch the installed b12x and quack packages to run on Windows (see below). |
| `sysmem_fallback.py` | Read or set the driver's "CUDA - Sysmem Fallback Policy" for `python.exe`. |
| `fallback_test.py` | Allocate past free VRAM to show whether the driver spills to system memory. |
| `chat_template.jinja` | The model's template, but accepting system and developer messages anywhere in a conversation. `chat_template.orig.jinja` is the stock one. |
| `verify_chat_template.py` | Check that the template changes nothing else. |
| `req_sequence.py` | Smoke test: short and code requests with and without thinking, with tokens/s. |
| `req_system_positions.py` | Live check that late system and developer messages are accepted and followed. |
| `bench_depth.py` | Decode speed at a fixed context depth with 1..N parallel requests. |

## Setup

1. Build and install vLLM from this branch into `.venv` at the repo root,
   following the main [README](../README.md); `buildenv.cmd` sets up the
   environment for that.
2. Patch b12x and quack. Both use `fcntl`, which Windows lacks; the patch falls
   back to `winflock.py`. It also fixes b12x's Triton hook rejecting the
   `FutureKernel` objects that Triton's parallel compile returns. Rerun after
   reinstalling either package:

   ```
   .venv\Scripts\python.exe qwen38-flash-next\patch_fcntl.py .venv\Lib\site-packages
   ```

3. Turn off the driver's sysmem fallback for `python.exe`, from an elevated
   prompt:

   ```
   .venv\Scripts\python.exe qwen38-flash-next\sysmem_fallback.py --apply
   ```

   With fallback on, WDDM backs VRAM overflow with system memory instead of
   failing the allocation, so PyTorch never releases its cache and memory use
   balloons while b12x repacks the MoE weights at load time. `--undo` reverts.
   Driver reinstalls may reset it; run without options to check.

4. Download the model into the standard Hugging Face cache.

## Run

```
qwen38-flash-next\runenv.cmd qwen38-flash-next\serve.cmd
```

The OpenAI-compatible API is then at `http://127.0.0.1:8000/v1`, and
`req_sequence.py` checks that it works. Leave plenty of free
system RAM: a 29 GB embedding table stays pinned there, and the 72 GB of weights in VRAM stream in through the file cache.

## Differences from the model card

- **Vision encoder on Humming, not b12x.** b12x's W4A16 linear kernel needs
  K to be a multiple of 32, and the vision encoder has a 4304-wide layer.
  `--kernel-config` routes only the W4A16 layers to Humming; the language
  model, MTP and MoE layers stay on b12x.
- **Linux-only or multi-GPU settings dropped:** `NCCL_*`, PCIe all-reduce,
  `B12X_DISK_BACKEND=io_uring`, `expandable_segments`,
  `--load-format instanttensor`, and variables that exist only in the
  community fork the card was written for.
- **`QUACK_ASYNC_COMPILE_START=spawn`,** because Windows has no `forkserver`.
- **Custom chat template,** so clients that send system or developer messages
  mid-conversation don't get an error.

## Known issues

- The first prefill of a new size JIT-compiles kernels, which adds a
  one-time latency spike.
- The Responses API rejects custom tools with a grammar format, so Codex has
  to be configured without its grammar-based `apply_patch` tool.
