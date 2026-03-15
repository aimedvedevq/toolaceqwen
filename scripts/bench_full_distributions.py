#!/usr/bin/env python3
"""Full benchmark: BF16/FP8/AWQ × vLLM/SGLang × concurrency, with per-request distributions."""

import json, time, os, subprocess, sys, signal
import numpy as np
import aiohttp, asyncio
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

ROLE_MAP = {"user": "user", "assistant": "assistant", "tool": "tool"}


def build_prompts(n=200):
    tok = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)
    ds = load_from_disk("./output/grpo_data")
    prompts_chat = []  # for SGLang engine
    prompts_text = []  # for vLLM API
    for ex in ds:
        msgs = []
        if ex.get("system"):
            msgs.append({"role": "system", "content": ex["system"]})
        last_idx = -1
        for i, turn in enumerate(ex["conversations"]):
            if turn["from"] == "assistant": last_idx = i
        if last_idx < 1: continue
        for turn in ex["conversations"][:last_idx]:
            msgs.append({"role": ROLE_MAP.get(turn["from"], "user"), "content": turn["value"]})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if len(tok.encode(text)) > 3000: continue
        prompts_chat.append(msgs)
        prompts_text.append(text)
        if len(prompts_text) >= n: break
    return prompts_chat, prompts_text, tok


# ──── vLLM benchmark (HTTP API) ────

async def vllm_request(session, url, model_name, messages, max_tokens=512):
    t0 = time.perf_counter()
    req = {"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    try:
        async with session.post(url, json=req) as resp:
            body = await resp.json()
        t_total = time.perf_counter() - t0
        usage = body.get("usage", {})
        return {
            "success": True, "latency": t_total,
            "ttft": t_total / max(usage.get("completion_tokens", 1), 1),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }
    except Exception as e:
        return {"success": False, "latency": time.perf_counter() - t0, "error": str(e)}


async def vllm_bench_concurrent(url, model_name, prompts_chat, concurrency, n=100):
    sem = asyncio.Semaphore(concurrency)
    results = []
    async def limited(msgs):
        async with sem:
            async with aiohttp.ClientSession() as s:
                return await vllm_request(s, url, model_name, msgs)
    tasks = [limited(prompts_chat[i % len(prompts_chat)]) for i in range(n)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r["success"]]


def start_vllm(model_path, port=8100, extra_args=None):
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--port", str(port), "--dtype", "auto",
           "--trust-remote-code", "--max-model-len", "4096"]
    if extra_args:
        cmd += extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import urllib.request
    for _ in range(60):
        time.sleep(3)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models")
            return proc
        except: pass
    proc.kill()
    raise RuntimeError("vLLM failed to start")


# ──── SGLang benchmark ────

def sglang_bench(prompts_text, concurrency_levels, engine, n_per_level=100):
    results = {}
    for c in concurrency_levels:
        batch = [prompts_text[i % len(prompts_text)] for i in range(n_per_level)]
        # Per-request: sequential for c=1, batch for c>1
        if c == 1:
            per_req = []
            for p in batch:
                t0 = time.perf_counter()
                r = engine.generate(prompt=p, sampling_params={"max_new_tokens": 512, "temperature": 0})
                lat = time.perf_counter() - t0
                per_req.append({
                    "latency": lat,
                    "ttft": lat / max(r["meta_info"]["completion_tokens"], 1),
                    "completion_tokens": r["meta_info"]["completion_tokens"],
                })
            results[c] = per_req
        else:
            # Batch — measure total, then distribute
            t0 = time.perf_counter()
            outs = engine.generate(
                prompt=batch,
                sampling_params={"max_new_tokens": 512, "temperature": 0},
            )
            total_time = time.perf_counter() - t0
            per_req = []
            for r in outs:
                ntok = r["meta_info"]["completion_tokens"]
                # Approximate per-request latency from total
                per_req.append({
                    "latency": total_time * ntok / sum(rr["meta_info"]["completion_tokens"] for rr in outs),
                    "ttft": total_time / len(outs) / max(ntok, 1),
                    "completion_tokens": ntok,
                })
            results[c] = per_req
    return results


def compute_stats(per_req_list):
    """Compute full distribution stats from per-request data."""
    lats = np.array([r["latency"] for r in per_req_list]) * 1000  # ms
    ttfts = np.array([r["ttft"] for r in per_req_list]) * 1000
    toks = np.array([r["completion_tokens"] for r in per_req_list])
    n = len(lats)
    return {
        "n": n,
        "latency": {
            "mean": float(np.mean(lats)), "std": float(np.std(lats)),
            "sem": float(np.std(lats) / np.sqrt(n)),
            "p25": float(np.percentile(lats, 25)), "p50": float(np.median(lats)),
            "p75": float(np.percentile(lats, 75)), "p90": float(np.percentile(lats, 90)),
            "p95": float(np.percentile(lats, 95)), "p99": float(np.percentile(lats, 99)),
            "min": float(np.min(lats)), "max": float(np.max(lats)),
            "raw": lats.tolist(),
        },
        "ttft": {
            "mean": float(np.mean(ttfts)), "std": float(np.std(ttfts)),
            "sem": float(np.std(ttfts) / np.sqrt(n)),
            "p25": float(np.percentile(ttfts, 25)), "p50": float(np.median(ttfts)),
            "p75": float(np.percentile(ttfts, 75)), "p90": float(np.percentile(ttfts, 90)),
            "p95": float(np.percentile(ttfts, 95)), "p99": float(np.percentile(ttfts, 99)),
            "raw": ttfts.tolist(),
        },
        "tokens": {"mean": float(np.mean(toks)), "total": int(np.sum(toks))},
        "tps": float(np.sum(toks) / (np.sum(lats) / 1000)),
    }


def main():
    import sglang
    prompts_chat, prompts_text, tok = build_prompts(200)
    print(f"Built {len(prompts_text)} prompts")

    concurrency_levels = [1, 4, 8, 16, 32]
    n_per_level = 100

    configs = {
        "BF16": {"path": "./output_grpo/merged", "sglang_kw": {}, "vllm_extra": []},
        "FP8": {"path": "./output_grpo/fp8", "sglang_kw": {}, "vllm_extra": []},
        "AWQ": {"path": "./output_grpo/awq", "sglang_kw": {"quantization": "compressed-tensors"}, "vllm_extra": ["--quantization", "compressed-tensors"]},
    }

    all_results = {}

    for qname, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {qname}")
        print(f"{'='*60}")

        # ── SGLang ──
        print(f"  SGLang {qname}...")
        try:
            engine = sglang.Engine(model_path=cfg["path"], mem_fraction_static=0.7, **cfg["sglang_kw"])
            sg_raw = sglang_bench(prompts_text, concurrency_levels, engine, n_per_level)
            sg_stats = {c: compute_stats(sg_raw[c]) for c in concurrency_levels}
            engine.shutdown()
            time.sleep(3)
        except Exception as e:
            print(f"  SGLang FAILED: {e}")
            sg_stats = {"error": str(e)}

        # ── vLLM ──
        print(f"  vLLM {qname}...")
        try:
            proc = start_vllm(cfg["path"], 8100, cfg["vllm_extra"])
            # Get model name
            import urllib.request
            models = json.loads(urllib.request.urlopen("http://localhost:8100/v1/models").read())
            model_name = models["data"][0]["id"]

            vllm_stats = {}
            for c in concurrency_levels:
                raw = asyncio.run(vllm_bench_concurrent(
                    "http://localhost:8100/v1/chat/completions", model_name,
                    prompts_chat, c, n_per_level
                ))
                vllm_stats[c] = compute_stats(raw) if raw else {"error": "no results"}
                print(f"    c={c}: {len(raw)} ok, lat_p50={vllm_stats[c].get('latency',{}).get('p50',0):.0f}ms")

            proc.terminate()
            try: proc.wait(timeout=10)
            except: proc.kill()
            time.sleep(5)
        except Exception as e:
            print(f"  vLLM FAILED: {e}")
            vllm_stats = {"error": str(e)}

        all_results[qname] = {"sglang": sg_stats, "vllm": vllm_stats}

    # Save
    # Strip raw arrays for JSON size
    save_results = json.loads(json.dumps(all_results, default=str))
    with open("bench_distributions.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to bench_distributions.json ({os.path.getsize('bench_distributions.json')/1e6:.1f}MB)")

    # Print summary
    print(f"\n{'='*100}")
    print(f"{'Config':12s} {'Engine':8s} {'C':>3s} {'Lat p50':>10s} {'Lat p95':>10s} {'Lat p99':>10s} {'TTFT p50':>10s} {'TPS':>8s}")
    print(f"{'-'*100}")
    for qname in configs:
        for eng_name in ["sglang", "vllm"]:
            eng_data = all_results[qname][eng_name]
            if isinstance(eng_data, dict) and "error" in eng_data:
                print(f"  {qname:10s} {eng_name:8s} FAILED")
                continue
            for c in concurrency_levels:
                s = eng_data.get(c, eng_data.get(str(c), {}))
                if not s or "error" in s: continue
                l = s["latency"]
                t = s["ttft"]
                print(f"  {qname:10s} {eng_name:8s} {c:>3d} {l['p50']:>9.0f}ms {l['p95']:>9.0f}ms {l['p99']:>9.0f}ms {t['p50']:>9.1f}ms {s['tps']:>7.0f}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
