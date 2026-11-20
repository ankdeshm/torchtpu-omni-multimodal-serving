#!/usr/bin/env python3
"""Dual-engine TorchTPU Omni / Multimodal serving benchmark harness.

Benchmarks OpenAI-compatible streaming chat endpoints (text and image-text multimodal)
served by either `vllm-torchtpu` (`v0.33.0` / `v0.32.0`) or `sglang-torchtpu`.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def make_test_image_data_url(width: int = 512, height: int = 512) -> str:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(72, 130, 185))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_prompt_text(approx_tokens: int) -> str:
    unit = "Analyze physical mechanics, visual layout, spatial relationships, and step-by-step reasoning across multimodal scenes. "
    reps = max(1, approx_tokens // 18)
    return (unit * reps)[: approx_tokens * 4]


def run_concurrency_bench(
    base_url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    num_requests: int,
    multimodal: bool,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    prompt_text = build_prompt_text(input_tokens)
    img_url = make_test_image_data_url(512, 512) if multimodal else None

    def send_one() -> tuple[float, float, int]:
        if multimodal:
            content: Any = [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": prompt_text},
            ]
        else:
            content = prompt_text
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        ttft = None
        comp_tokens = 0
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:].strip())
                choices = ev.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                        ttft = time.perf_counter() - t0
                if ev.get("usage"):
                    comp_tokens = int(ev["usage"].get("completion_tokens", comp_tokens))
        total = time.perf_counter() - t0
        return (ttft or total), total, max(comp_tokens, 1)

    wall0 = time.perf_counter()
    ttfts_ms, tpots_ms, total_toks = [], [], 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(send_one) for _ in range(num_requests)]
        for fut in as_completed(futs):
            ttft_s, tot_s, toks = fut.result()
            ttfts_ms.append(ttft_s * 1000.0)
            tpots_ms.append(((tot_s - ttft_s) * 1000.0) / max(toks - 1, 1))
            total_toks += toks
    wall_s = time.perf_counter() - wall0
    return {
        "model": model,
        "multimodal": multimodal,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "output_tok_s": round(total_toks / wall_s, 2),
        "ttft_p50_ms": round(statistics.median(ttfts_ms), 2),
        "tpot_p50_ms": round(statistics.median(tpots_ms), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TorchTPU Omni multimodal serving.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    res = run_concurrency_bench(
        args.base_url,
        args.model,
        args.input_tokens,
        args.output_tokens,
        args.concurrency,
        args.num_requests,
        args.multimodal,
    )
    print(json.dumps(res, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
