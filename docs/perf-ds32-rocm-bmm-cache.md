# DeepSeek V3.2 ROCm Performance Optimization: BF16 Weight Caching & FP8 BMM Enablement

## Summary

Performance optimizations for DeepSeek V3.2 serving on AMD ROCm (MI300X / GFX942), targeting the MLA absorbed attention BMM paths. Two files modified:

- `forward_mla.py` — main MLA absorbed forward path
- `forward_mla_fused_rope_rocm.py` — fused RoPE + MLA decode path (ROCm-specific)

## Optimizations

### 1. Cached BF16 Weight Matrices (`forward_mla.py` + `forward_mla_fused_rope_rocm.py`)

The MLA absorbed attention runs two batched matrix multiplies per layer: `q_nope @ w_kc` (query absorption) and `attn_output @ w_vc` (output absorption). On the BF16 fallback path (entered when `_is_hip` is true and weights are not FP8/MXFP4), each call evaluated `self.w_kc.to(bf16) * self.w_scale` inline. Even for BF16 models where `w_scale = 1.0`, the `* w_scale` creates a temporary allocation every call — repeated 122 times per decode step (61 layers × 2 BMMs). Each temporary is `(num_heads, K, N)` at BF16 ≈ 2 MB, totaling ~244 MB of transient memory churn per step. The optimized code caches these as `self._w_kc_bf16` / `self._w_vc_bf16` on first use.

### 2. FP8 BMM Support in Fused RoPE Path (`forward_mla_fused_rope_rocm.py`)

The fused RoPE decode path previously used a plain BF16 BMM fallback for all HIP cases. Now it matches the optimized `forward_mla.py` logic with three tiers:
- **MXFP4 (GFX950):** `batched_gemm_afp4wfp4_pre_quant` for `uint8` weights
- **FP8 (GFX942 + GFX950):** `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant` for `float8_e4m3fn` (gfx950, always) / `float8_e4m3fnuz` (gfx942, CUDA-graph mode only — hides triton launch overhead)
- **BF16 fallback:** cached weight conversion (optimization #1)

### 3. Fused Output Quantization in Fused RoPE Path (`forward_mla_fused_rope_rocm.py`)

When `o_proj` weights are quantized (MXFP4 or FP8), the output of the `w_vc` BMM is now fused with quantization via `fused_flatten_mxfp4_quant` or `fused_flatten_fp8_group_quant`, eliminating a separate kernel launch. GFX950 only.

## Hardware Applicability

| Optimization | GFX942 | GFX950 |
|---|:-:|:-:|
| Cached BF16 weights | ✅ | ✅ |
| FP8 BMM (`batched_gemm_a8w8`) | ✅ (fnuz dtype) | ✅ (e4m3fn dtype) |
| MXFP4 BMM (`batched_gemm_afp4wfp4`) | ❌ | ✅ |
| Fused output quant | ❌ | ✅ |

## Benchmark Results

**Hardware:** AMD Instinct MI300X (GFX942), 8 GPUs
**Model:** `deepseek-ai/DeepSeek-V3.2` (BF16 weights)

### Random Serving (256 prompts, 8K input / 2K output, rate=∞)

| Metric | TP8 Before | TP8 After | **Δ** | TP8+DP8 Before | TP8+DP8 After | **Δ** |
|---|--:|--:|--:|--:|--:|--:|
| Duration (s) | 260.32 | 256.76 | **−1.4%** | 264.91 | 241.40 | **−8.9%** |
| Request throughput (req/s) | 0.98 | 1.00 | +2.0% | 0.97 | 1.06 | **+9.3%** |
| Input tok/s | 3,998.8 | 4,054.2 | +1.4% | 3,929.6 | 4,312.3 | **+9.7%** |
| Output tok/s | 972.7 | 986.2 | +1.4% | 955.9 | 1,049.0 | **+9.7%** |
| Peak output tok/s | 2,210 | 2,259 | +2.2% | 2,711 | 2,770 | +2.2% |
| **Total tok/s** | **4,971.5** | **5,040.5** | **+1.4%** | **4,885.4** | **5,361.3** | **+9.7%** |
| Concurrency | 179.6 | 180.3 | — | 178.0 | 174.5 | — |
| Mean E2E (ms) | 182,676 | 180,819 | −1.0% | 184,235 | 164,550 | **−10.7%** |
| Median E2E (ms) | 186,077 | 184,216 | −1.0% | 185,153 | 165,543 | **−10.6%** |
| P90 E2E (ms) | 244,066 | 241,086 | −1.2% | 248,867 | 226,315 | **−9.1%** |
| P99 E2E (ms) | 254,905 | 251,539 | −1.3% | 263,063 | 239,690 | **−8.9%** |
| Mean TTFT (ms) | 45,389 | 45,124 | −0.6% | 43,022 | 34,876 | **−18.9%** |
| Median TTFT (ms) | 42,437 | 42,094 | −0.8% | 39,511 | 32,327 | **−18.2%** |
| P99 TTFT (ms) | 110,859 | 110,059 | −0.7% | 90,639 | 75,913 | **−16.2%** |
| Mean TPOT (ms) | 193.1 | 191.3 | −0.9% | 212.1 | 190.2 | **−10.3%** |
| Median TPOT (ms) | 140.5 | 138.9 | −1.2% | 145.2 | 132.6 | **−8.7%** |
| P99 TPOT (ms) | 1,277 | 1,269 | −0.6% | 1,436 | 1,276 | **−11.1%** |
| Mean ITL (ms) | 139.5 | 137.9 | −1.1% | 142.9 | 131.3 | **−8.2%** |
| Median ITL (ms) | 101.8 | 100.2 | −1.6% | 92.6 | 88.7 | **−4.2%** |
| P95 ITL (ms) | 111.8 | 110.4 | −1.2% | 107.2 | 105.1 | −2.1% |
| P99 ITL (ms) | 437.3 | 433.7 | −0.8% | 603.9 | 515.5 | **−14.7%** |
| Max ITL (ms) | 73,595 | 72,970 | −0.8% | 86,881 | 75,309 | **−13.3%** |

### GSM8K Few-Shot (200 questions, 5-shot, max 512 tokens, parallel=128)

| Metric | TP8 Before | TP8 After | **Δ** | TP8+DP8 Before | TP8+DP8 After | **Δ** |
|---|--:|--:|--:|--:|--:|--:|
| Output tok/s | 544.5 | 706.3 | **+29.7%** | 358.5 | 469.4 | **+30.9%** |
| Latency (s) | 32.87 | 25.79 | **−21.5%** | 50.81 | 39.29 | **−22.7%** |
| Accuracy | 0.975 | 0.950 | ±noise | 0.960 | 0.970 | ±noise |

### Key Observations

- **GSM8K ~30% throughput gain** across both configs — decode-dominated workload amplifies per-step savings
- **TP8+DP8 ~10% serving throughput, 18% TTFT improvement** — 8 model replicas multiply the eliminated allocations (976 per step)
- **TP8-only ~1–2% serving improvement** — single replica, large batches, allocation overhead is smaller fraction
- **No accuracy regression** — fluctuations within statistical noise for 200 questions

## Reproducing

### Launch Server

```bash
# TP8 only
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2 --tp 8

# TP8 + DP8
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2 --tp 8 --dp 8 --enable-dp-attention
```

### Random Serving Benchmark

```bash
python3 -m sglang.bench_serving \
  --model deepseek-ai/DeepSeek-V3.2 \
  --backend vllm --host 0.0.0.0 \
  --dataset-name "random" \
  --random-input-len 8192 --random-output-len 2048 \
  --num-prompts 256 --request-rate "inf" \
  --seed 1 --port 30000
```

### GSM8K Accuracy + Throughput

```bash
python -m sglang.test.few_shot_gsm8k \
  --num-shots 5 --num-questions 200 \
  --max-new-tokens 512 --parallel 128 \
  --host http://127.0.0.1 --port 30000
```
