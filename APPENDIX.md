# Appendix — Omni & Multimodal Serving Ablations, Engine Comparisons & Failure Diagnostics

This appendix records full quantization ablations (`BF16` vs. `KV-FP8` vs. `experts` Quantize-on-Load), `TP=4` vs. `TP=8` scaling, `vllm-torchtpu v0.32.0` vs. `v0.33.0` upgrades, and exact `sglang-torchtpu` compatibility diagnostics across `Qwen3-Omni-30B-A3B` and `Ming-flash-omni-2.0`.

---

## 1. `Qwen3-Omni-30B-A3B`: Precision & Tensor Parallelism Ablations

### TPU v7x: `TP=4` (2 chips) vs. `TP=8` (4 chips), `BF16` vs. `KV-FP8` vs. `experts` QOL

| Engine & Config | Chips (`TP`) | `1k/256 c=1` | `1k/256 c=32` | `1k/256 c=64` | `8k/512 c=16` | `mm_img c=16` | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vllm v0.33.0` `BF16`** | 2 (`TP=4`) | 170.2 tok/s (`38.9 ms` TTFT) | 1,448.7 tok/s | 1,620.9 tok/s | 1,002.6 tok/s | **647.3 tok/s** | Native multimodal + `32K` ctx (`235.3 tok/s` at `c=4`) |
| `vllm v0.33.0` `experts` QOL + `KV-FP8` | 2 (`TP=4`) | — | — | — | 982.4 tok/s (`15.3 ms` TPOT) | — | Quantize-on-load (`QUANTIZE_ON_LOAD_PREFIXES=experts`) saves HBM with ~2% throughput delta |
| `vllm v0.32.0` `BF16` | 2 (`TP=4`) | 183.6 tok/s (`39.3 ms` TTFT) | 1,493.6 tok/s | — | 1,007.0 tok/s | *Needs v0.33* | Highest text throughput per chip (`810 tok/s/chip` at `c=64` on `v0.33`) |
| `vllm v0.32.0` `BF16` | 4 (`TP=8`) | 188.1 tok/s (`36.5 ms` TTFT) | 1,505.0 tok/s | **1,674.2 tok/s** | — | *Needs v0.33* | Peak aggregate throughput (+3.3% over `TP=4`) |
| **`vllm v0.32.0` `KV-FP8`** | 4 (`TP=8`) | **195.8 tok/s (`34.8 ms` TTFT)** | — | **1,673.8 tok/s** | 1,002.2 tok/s | *Needs v0.33* | Lowest `c=1` latency (`4.86 ms` TPOT); matches `BF16` at `c=64` |

### TPU v5p (`TP=4`, 4 chips): `BF16` vs. `KV-FP8` vs. `experts` QOL

| Variant & Engine | Precision | `1k/256 (or 512) c=1` | `c=32` | `c=64` | `8k/512 c=16` | `32k/256 c=4` | `mm_img c=16` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Instruct` (`vllm v0.33.0`) | `BF16` | 128.4 tok/s | 1,080.3 tok/s | **1,290.6 tok/s** | **738.5 tok/s** | 135.6 tok/s | **482.3 tok/s** |
| `Instruct` (`vllm v0.33.0`) | `experts` QOL + `KV-FP8` | — | — | — | 732.4 tok/s | — | — |
| `Instruct` (`vllm v0.32.0`) | `BF16` | 134.9 tok/s | 1,116.7 tok/s | **1,321.6 tok/s** | — | — | *Needs v0.33* |
| `Instruct` (`vllm v0.32.0`) | `KV-FP8` | 130.7 tok/s | 998.3 tok/s | 1,167.4 tok/s | 678.0 tok/s | **204.2 tok/s** | — |
| `Thinking` (`vllm v0.32.0`) | `BF16` | **141.6 tok/s** | **1,229.7 tok/s** | — | **766.8 tok/s** | — | — |
| `Thinking` (`vllm v0.32.0`) | `KV-FP8` | 136.1 tok/s | 1,053.0 tok/s | — | 710.5 tok/s | **220.8 tok/s** | — |

**Key `KV-FP8` finding on TPU v5p:** On `Qwen3-Omni-30B-A3B`, `KV-FP8` reduced short-context throughput by ~12–14% on TPU v5p (`1,167.4` vs `1,321.6 tok/s` at `c=64`) while improving `32K` long-context throughput (`204.2–220.8 tok/s` at `c=4` vs `135.6 tok/s` `BF16`). On TPU v7x (`TP=8`), `KV-FP8` improved `c=1` decode speed (`195.8` vs `188.1 tok/s`) with no throughput loss at `c=64`.

---

## 2. `sglang-torchtpu` Dual-Engine Audit & Upstream Bug Diagnostics

1. **`Qwen3-Omni-30B-A3B` on `sglang-torchtpu v0.2.0` (eager mode, TPU v5p `TP=4`):**
   - Verified end-to-end text (`1k`, `8k`, `32k`) and multimodal image-text (`mm_img_512_256`) serving on both `Instruct` (`146.5 tok/s` at `c=32`, `64.1 tok/s` multimodal at `c=16`) and `Thinking` (`155.3 tok/s` at `c=32`).
   - Because TPU graph compilation was disabled in eager mode, throughput was `~8.8×` lower than compiled `vllm-torchtpu` (`1,290.6 tok/s`) on the same 4-chip TPU v5p hardware.
2. **`Qwen3-Omni-30B-A3B` on `sglang-torchtpu` (`main`, compiled mode, TPU v7x `TP=4`):**
   - Successfully precompiled all 29 vision patch buckets and all 6 DECODE TPU graphs (`78.2 / 97.0 GB` HBM), then exited during scheduler initialization with upstream error:
     `AttributeError: 'Qwen3OmniMoeForConditionalGeneration' object has no attribute 'capture_aux_hidden_states'`
   - **Recommendation:** Use **`vllm-torchtpu v0.33.0`** for compiled production serving of `Qwen3-Omni-30B-A3B` (text + multimodal) until this upstream attribute check is resolved in `sglang-torchtpu`.
3. **`Ming-flash-omni-2.0` on `sglang-torchtpu` (`main`):**
   - Failed during `ModelConfig._derive_context_length` (`sglang/srt/hf_transformers/common.py:703`) with `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` because `rope_scaling_factor` is `None` in the model config.
   - Use `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`deploy/ming_flash_omni.yaml`) instead.

---

## 3. Why `vllm-torchtpu v0.33.0` Matters for Multimodal Serving (and Mixed Audio+Image Boundary)

On `vllm-torchtpu v0.32.0`, multimodal image-text requests to `Qwen3-Omni-30B-A3B` failed prior to the upstream multimodal fix (`PR #2993`). Upgrading to **`vllm-torchtpu v0.33.0`** enables native **image**, **audio** (`5s`/`15s` WAV up to `761.7 tok/s`), and **video** (`video_url` MP4 up to `376.3 tok/s`, 8-frame sequences up to `507.7 tok/s`) serving out of the box across TPU v7x and TPU v5p.

- **Upstream boundary — combined `audio_url` + `image_url` in the same request:** Sending both an audio clip and an image inside a single `/v1/chat/completions` message currently fails in `vllm v0.33.0` EngineCore with:
  `RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 2048 but got size 8192 for tensor number 1 in the list.`
  This occurs because `Qwen3-Omni` concatenates 4-level DeepStack visual features (`4 × 2048 = 8192`) alongside standard `2048`-dim audio features in the multimodal feature list before language-model injection. Single-modality requests (`audio`, `image`, or `video` individually) do not trigger this path and run cleanly.
