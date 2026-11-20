# Working Log — Omni & Multimodal Serving on TorchTPU

## Goal
Evaluate serving throughput, latency, quantization options (`BF16`, `KV-FP8`, `experts` quantize-on-load), and multimodal image-text support for omni-modal models on Google Cloud TPU v7x and TPU v5p across `vllm-torchtpu` and `sglang-torchtpu`.

## What We Tried & Found

### 1. `Qwen3-Omni-30B-A3B` on `vllm-torchtpu v0.32.0` vs. `v0.33.0`
- **Issue on `v0.32.0`:** Text serving reached `1,674.2 tok/s` (`TP=8`, `c=64`) and `195.8 tok/s` `c=1` (`KV-FP8`) on TPU v7x, and `1,229.7 tok/s` per 4-chip replica (`~9,838 tok/s` across 8 replicas on a 32-chip slice) on TPU v5p, but multimodal image-text requests failed without upstream multimodal fixes.
- **Resolution on `v0.33.0`:** Upgrading to `vllm-torchtpu v0.33.0` enabled image-text serving (`mm_img_512_256`) out of the box at `647.3 tok/s` (`c=16`, `55.3 ms` TTFT at `c=4`) on 2 TPU v7x chips (`TP=4`) and `482.3 tok/s` on 4 TPU v5p chips (`TP=4`), while sustaining `1,620.9 tok/s` text throughput (`c=64`) on 2 TPU v7x chips (`1.94×` higher throughput per chip than `TP=8`).
- **Quantization trade-offs:** `KV-FP8` improved `c=1` decode latency on TPU v7x `TP=8` (`195.8` vs. `188.1 tok/s`, `34.8 ms` TTFT) and lifted `32K` long-context throughput on TPU v5p (`204.2–220.8 tok/s` vs. `135.6 tok/s` `BF16`), while slightly lowering short-context throughput on TPU v5p (`-12%`). Combining `QUANTIZE_ON_LOAD_PREFIXES=experts` with `KV-FP8` on `v0.33.0` stayed within ~2% of `BF16` at `8k/512 c=16` (`982.4` vs. `1,002.6 tok/s` on TPU v7x).

### 2. `sglang-torchtpu` Bring-Up & Open Issues
- **Eager mode (`v0.2.0`, TPU v5p `TP=4`):** Verified text (`1k`, `8k`, `32k`) and image-text serving on both `Instruct` (`146.5 tok/s` at `c=32`, `64.1 tok/s` multimodal) and `Thinking` (`155.3 tok/s`), ~8.8× slower than compiled `vllm-torchtpu`.
- **Compiled mode (`main`, TPU v7x `TP=4`):** Precompiled all 29 vision buckets and 6 decode graphs (`78.2 / 97.0 GB` HBM), then exited with `AttributeError: 'Qwen3OmniMoeForConditionalGeneration' object has no attribute 'capture_aux_hidden_states'` — an open upstream bug in `sglang-torchtpu`.

### 3. `Ming-flash-omni-2.0` Bring-Up Findings
- **`vllm-omni 0.33.0rc1` (`deploy/ming_flash_omni.yaml`):** Loading all 42 safetensors shards (~200 GB) from cold object storage exceeded the default `--stage-init-timeout 300`. Passing `--stage-init-timeout 2400` allowed all 42 shards and 32 `PallasAttentionBackendImpl` layers to load and complete `torch.compile` in `51.08 s` on TPU v7x, after which `_dummy_run()` (`vllm_torchtpu/runner/tpu_runner.py:6406`) raised `TypeError: expected torch.Tensor` inside `synchronize_tensors(out)` — an open upstream bug in `vllm-torchtpu v0.33.0`.
- **`sglang-torchtpu` (`main`):** Exits during `ModelConfig._derive_context_length` (`TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`) because `rope_scaling_factor` is `None` in the model's config.
