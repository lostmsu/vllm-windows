@echo off
rem Qwen3.8-Flash-Next-FP4-Local on one RTX PRO 6000 Blackwell, Windows, vLLM 0.34.
rem Adapted from the model card's PRO 6000 recipe. Dropped as not applicable here:
rem   NCCL_*, PCIe all-reduce (TP=1), B12X_DISK_BACKEND=io_uring (Linux),
rem   expandable_segments (unsupported on Windows), --load-format instanttensor
rem   (libaio, Linux-only), and community-fork-only VLLM_* variables.
rem   --gdn-decode-kernel b12x has no upstream flag; upstream uses its fused CUDA
rem   GDN decode kernel (VLLM_GDN_DECODE_KERNEL=cuda, the default).
cd /d %~dp0
set B12X_DENSE_SPLITK_TURBO=1
set B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0
set B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1
set B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1
set B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224
set B12X_DYNAMIC_SPLIT_FAST_PREPARE=1
set B12X_DYNAMIC_SPLIT_LOW_SMEM=1
set B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1
set B12X_DYNAMIC_WORK_SOURCE=persistent_grid
set B12X_W4A16_A4_PREFILL=1
set B12X_MHC_PDL=1
set CUDA_BINARY_LOADER_THREAD_COUNT=8
set CUDA_MODULE_LOADING=LAZY
set CUTE_DSL_ARCH=sm_120a
set OMP_NUM_THREADS=2
rem Windows has no forkserver start method
set QUACK_ASYNC_COMPILE_START=spawn
set PYTHONFAULTHANDLER=1
set TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0
set TORCH_CUDA_ARCH_LIST=12.0a
set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
set VLLM_B12X_MOE_FP4_FORCE_A16=1
set VLLM_COMPUTE_NANS_IN_LOGITS=0
set VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
set VLLM_SSM_CONV_STATE_LAYOUT=DS
set VLLM_USE_FASTOKENS=1
rem chat_template.jinja: the stock template (chat_template.orig.jinja), but system
rem and developer messages are accepted anywhere and kept in place, where the
rem stock one rejects any after the first (check: verify_chat_template.py).
if "%GPU_UTIL%"=="" set GPU_UTIL=0.88
if "%SERVE_HOST%"=="" set SERVE_HOST=127.0.0.1
if "%SERVE_PORT%"=="" set SERVE_PORT=8000
vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local ^
  --host %SERVE_HOST% --port %SERVE_PORT% ^
  --async-scheduling ^
  --block-size 64 ^
  --chat-template "%~dp0chat_template.jinja" ^
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" ^
  --decode-context-parallel-size 1 ^
  --dtype bfloat16 ^
  --enable-auto-tool-choice ^
  --enable-chunked-prefill ^
  --no-enable-flashinfer-autotune ^
  --enable-prefix-caching ^
  --enable-prompt-tokens-details ^
  --gpu-memory-utilization %GPU_UTIL% ^
  --kv-cache-dtype fp8 ^
  --no-language-model-only ^
  --linear-backend b12x ^
  --kernel-config "{\"linear_backend_per_quant\":{\"nvfp4_w4a16\":\"humming\"}}" ^
  --mamba-cache-mode align ^
  --mamba-ssm-cache-dtype auto ^
  --max-cudagraph-capture-size 64 ^
  --max-model-len 262144 ^
  --max-num-batched-tokens 6019 ^
  --max-num-seqs 16 ^
  --mm-encoder-tp-mode data ^
  --mm-processor-cache-gb 0.0 ^
  --moe-backend b12x ^
  --pipeline-parallel-size 1 ^
  --quantization modelopt_mixed ^
  --reasoning-parser qwen3 ^
  --speculative-config "{\"method\":\"mtp\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"b12x\",\"num_speculative_tokens\":3}" ^
  --tensor-parallel-size 1 ^
  --tool-call-parser qwen3_xml %*
