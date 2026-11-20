# Ming-flash-omni-2.0 on Google Cloud TPUs (TorchTPU)

- **Model:** [`inclusionAI/Ming-flash-omni-2.0`](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) (100B-total / 6.1B-active MoE unified omni-modal model spanning text, image, audio, and vision generation; 42 safetensors shards, `bf16`)
- **License:** MIT
- **Engines Audited:** `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`deploy/ming_flash_omni.yaml`) and `sglang-torchtpu` (`main`)

## Integration & Startup Audit Findings

1. **`vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`, native `deploy/ming_flash_omni.yaml`):**
   - `vllm-omni 0.33.0rc1` ships a native multi-stage deploy YAML (`vllm_omni/deploy/ming_flash_omni.yaml`) for `inclusionAI/Ming-flash-omni-2.0`.
   - Loading all 42 safetensors shards (~200 GB) from network-mounted object storage exceeds the default `--stage-init-timeout 300`; passing `--stage-init-timeout 2400` allows all 42 shards and 32 `PallasAttentionBackendImpl` layers to load cleanly on TPU v7x and complete `torch.compile` AOT compilation in **`51.08 s`**.
   - Immediately after compilation, during `determine_available_memory()` -> `_dummy_run()` (`vllm_torchtpu/runner/tpu_runner.py:6406`), `synchronize_tensors(out)` raises:
     `TypeError: expected torch.Tensor`
     because `Ming-flash-omni-2.0`'s multi-output forward returns non-tensor elements that `vllm_torchtpu/utils.py:408` passes directly to `sync.synchronize(tensors=tensors)`.
2. **`sglang-torchtpu` (`main`) Compatibility Boundary:**
   - Launching `sglang serve` on `inclusionAI/Ming-flash-omni-2.0` exits during `ModelConfig._derive_context_length` (`sglang/srt/hf_transformers/common.py:703`) with:
     `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`
     when `rope_scaling_factor` is `None` in the checkpoint's HF text config.
   - Both upstream issues are documented here so teams evaluating `Ming-flash-omni-2.0` can track fixes in `vllm-torchtpu` and `sglang-torchtpu`; use **`Qwen3-Omni-30B-A3B`** on `vllm-torchtpu v0.33.0` (`1,620.9 tok/s` text / `647.3 tok/s` multimodal on 2 TPU v7x chips) for production omni-modal serving today.

## Reproduce

```bash
vllm serve /path/to/inclusionAI/Ming-flash-omni-2.0 \
  --omni \
  --stage-init-timeout 2400 \
  --deploy-config $(python3 -c 'import vllm_omni, os; print(os.path.join(os.path.dirname(vllm_omni.__file__), "deploy/ming_flash_omni.yaml"))') \
  --served-model-name inclusionAI/Ming-flash-omni-2.0 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code
```
