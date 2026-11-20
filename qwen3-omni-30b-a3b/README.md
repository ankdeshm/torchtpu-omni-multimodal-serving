# Qwen3-Omni-30B-A3B (`Instruct` & `Thinking`) on Google Cloud TPUs (TorchTPU)

- **Models:** [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) and [`Qwen/Qwen3-Omni-30B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) (30B total / 3B active MoE Thinker-Talker omni-modal model, 128 experts, `36K` context tested, `bf16` & `KV-FP8`)
- **License:** Apache-2.0
- **Engines Tested:** `vllm-torchtpu v0.33.0` (native multimodal), `vllm-torchtpu v0.32.0`, and `sglang-torchtpu` (`v0.2.0` eager & `main` compiled)

## Measured Results — `Qwen3-Omni-30B-A3B-Instruct`

| Engine & Precision | Hardware & Topology | Scenario (`in/out`) | Concurrency | Output Throughput | TTFT p50 | TPOT p50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | `text 1k/256` | 1 | **170.2 tok/s** | 38.9 ms | 5.11 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | `text 1k/256` | 32 | 1,448.7 tok/s | 508.3 ms | 20.12 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | `text 1k/256` | 64 | **1,620.9 tok/s** | 615.7 ms | 36.14 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | `text 8k/512` | 16 | 1,002.6 tok/s | 643.7 ms | 14.56 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | `text 32k/256` | 4 | **235.3 tok/s** | 396.3 ms | 9.26 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | **`mm_img 512×512 + 256`** | **4** | **393.1 tok/s** | **55.3 ms** | **6.99 ms** |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v7x (2 chips / `TP=4`)** | **`mm_img 512×512 + 256`** | **16** | **647.3 tok/s** | 1,452.8 ms | 12.73 ms |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v7x (4 chips / `TP=8`)** | `text 1k/256` | 64 | **1,674.2 tok/s** | 752.6 ms | 34.49 ms |
| **`vllm-torchtpu v0.32.0` (`KV-FP8`)** | **TPU v7x (4 chips / `TP=8`)** | `text 1k/256` | 1 | **195.8 tok/s** | **34.8 ms** | **4.86 ms** |
| **`vllm-torchtpu v0.32.0` (`KV-FP8`)** | **TPU v7x (4 chips / `TP=8`)** | `text 1k/256` | 64 | **1,673.8 tok/s** | 699.1 ms | 33.99 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v5p (4 chips / `TP=4`)** | `text 1k/256` | 64 | **1,290.6 tok/s** | 430.0 ms | 46.99 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v5p (4 chips / `TP=4`)** | **`mm_img 512×512 + 256`** | **4** | 306.6 tok/s | 145.3 ms | 9.95 ms |
| **`vllm-torchtpu v0.33.0` (`BF16`)** | **TPU v5p (4 chips / `TP=4`)** | **`mm_img 512×512 + 256`** | **16** | **482.3 tok/s** | 2,711.4 ms | 17.22 ms |
| `vllm-torchtpu v0.32.0` (`BF16`) | TPU v5p (4 chips / `TP=4`) | `text 1k/256` | 64 | 1,321.6 tok/s | 416.0 ms | 45.94 ms |
| `sglang-torchtpu v0.2.0` (eager) | TPU v5p (4 chips / `TP=4`) | `text 1k/256` / `mm_img` | 32 / 16 | 146.5 / 64.1 tok/s | 7,434 / 7,855 ms | 190 / 228 ms |

## Measured Results — `Qwen3-Omni-30B-A3B-Thinking`

| Engine & Precision | Hardware & Topology | Scenario (`in/out`) | Concurrency | Output Throughput | TTFT p50 | TPOT p50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v7x (4 chips / `TP=8`)** | `text 1k/512` | 1 | **199.0 tok/s** | **35.9 ms** | **4.90 ms** |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v7x (4 chips / `TP=8`)** | `text 1k/512` | 32 | **1,595.5 tok/s** | 268.5 ms | 19.43 ms |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v7x (4 chips / `TP=8`)** | `text 8k/512` | 16 | **1,107.4 tok/s** | 565.0 ms | 13.51 ms |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v5p (4 chips / `TP=4`)** | `text 1k/512` | 32 | **1,229.7 tok/s** | 271.7 ms | 25.37 ms |
| **`vllm-torchtpu v0.32.0` (`BF16`)** | **TPU v5p (4 chips / `TP=4`)** | `text 8k/512` | 16 | 766.8 tok/s | 887.0 ms | 19.38 ms |
| `vllm-torchtpu v0.32.0` (`KV-FP8`) | TPU v5p (4 chips / `TP=4`) | `text 1k/512` / `32k/256` | 32 / 4 | 1,053.0 / 220.8 tok/s | 401.8 / 628.6 ms | 29.58 / 10.66 ms |
| `sglang-torchtpu v0.2.0` (eager) | TPU v5p (4 chips / `TP=4`) | `text 1k/512` | 32 | 155.3 tok/s | 7,408.7 ms | 191.9 ms |

### Key Takeaways
- **`vllm-torchtpu v0.33.0` supports multimodal serving natively:** Image-text (`512×512`) requests run cleanly at **`647.3 tok/s`** (`c=16`) and **`393.1 tok/s` (`55.3 ms` TTFT at `c=4`)** on 2 TPU v7x chips (`TP=4`), `1.34×` faster than 4 TPU v5p chips (`482.3 tok/s`).
- **2 TPU v7x chips (`TP=4`) nearly saturate a 4-chip (`TP=8`) deployment:** On `1k/256 c=64`, `TP=4` on 2 v7x chips reaches `1,620.9 tok/s` vs. `1,674.2 tok/s` on 4 v7x chips (`TP=8`) — giving **1.94× higher throughput per chip** at `TP=4`.
- **Linear multi-replica scaling on TPU v5p-32:** Running 8 independent `TP=4` replicas across a 32-chip TPU v5p slice scaled `Qwen3-Omni-30B-A3B-Thinking` linearly to `~9,838 tok/s` aggregate.

## Reproduce

```bash
# Recommended: vllm-torchtpu v0.33.0 on 2 TPU v7x chips (TP=4) for text + multimodal
vllm serve /path/to/Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 36864 \
  --trust-remote-code

python3 ../scripts/benchmark_omni_serving.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-tokens 512 --output-tokens 256 --concurrency 16 --num-requests 16 --multimodal
```
