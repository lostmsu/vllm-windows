"""Measure decode speed at a fixed context depth with 1..N parallel requests.

Each request gets its own prompt of ~depth tokens, built from disjoint vLLM
source files so no request hits another's prefix cache, and asks for a long
explanation with thinking on. Aggregate decode speed is measured over the
window in which every request of a level is decoding, so prefill of the later
requests doesn't dilute it.

Usage: python bench_depth.py [--url URL] [--depth 102400] [--max-tokens 2048]
                             [--levels 1,2,3,4] [--src DIR]
"""
import argparse
import json
import re
import threading
import time
import urllib.request
from bisect import bisect_right
from pathlib import Path

MODEL = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
TASK = ("Explain in detail what each file above does and how the files relate "
        "to each other, file by file.")


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def post_json(base, path, body):
    with post(base, path, body) as r:
        return json.load(r)


def messages(doc):
    return [{"role": "user", "content": f"{doc}\n\n{TASK}"}]


class Docs:
    """Hands out documents of a fixed token length from disjoint files."""

    def __init__(self, base, src, depth):
        self.base = base
        self.root = Path(src)
        self.files = iter(sorted(self.root.rglob("*.py")))
        overhead = post_json(base, "/tokenize", {
            "model": MODEL, "messages": messages(""),
            "add_generation_prompt": True})["count"]
        self.doc_tokens = depth - overhead

    def _tokenize(self, text):
        return post_json(self.base, "/tokenize", {
            "model": MODEL, "prompt": text,
            "add_special_tokens": False})["tokens"]

    def next(self):
        text, need_chars = "", self.doc_tokens * 4.5
        while True:
            while len(text) < need_chars:
                f = next(self.files)
                rel = f.relative_to(self.root.parent).as_posix()
                body = f.read_text(encoding="utf-8", errors="replace")
                text += f"### {rel}\n{body}\n"
            tokens = self._tokenize(text)
            if len(tokens) >= self.doc_tokens:
                return post_json(self.base, "/detokenize", {
                    "model": MODEL,
                    "tokens": tokens[:self.doc_tokens]})["prompt"]
            need_chars = len(text) * self.doc_tokens / len(tokens) + 4000


def run_request(base, doc, max_tokens, start, out):
    body = {"model": MODEL, "messages": messages(doc),
            "max_tokens": max_tokens, "temperature": 0.6, "stream": True,
            "stream_options": {"include_usage": True,
                               "continuous_usage_stats": True}}
    times, counts = [], []
    out.update(times=times, counts=counts, finish=None, prompt_tokens=None)
    start.wait()
    out["t0"] = time.perf_counter()
    try:
        with post(base, "/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                t = time.perf_counter()
                chunk = json.loads(line[6:])
                usage = chunk.get("usage")
                if usage:
                    out["prompt_tokens"] = usage["prompt_tokens"]
                    n = usage.get("completion_tokens") or 0
                    if n and (not counts or n > counts[-1]):
                        times.append(t)
                        counts.append(n)
                for choice in chunk.get("choices") or []:
                    out["finish"] = choice.get("finish_reason") or out["finish"]
    except Exception as e:  # noqa: BLE001 - report and keep the other streams
        out["error"] = repr(e)


def spec_counters(base):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        text = r.read().decode()
    values = {}
    for name in ("drafts", "draft_tokens", "accepted_tokens"):
        m = re.search(
            rf"^vllm:spec_decode_num_{name}_total(?:\{{[^}}]*\}})? (\S+)$",
            text, re.M)
        values[name] = float(m.group(1)) if m else 0.0
    return values


def tokens_at(r, t):
    i = bisect_right(r["times"], t)
    return r["counts"][i - 1] if i else 0


def run_level(base, docs, n, max_tokens):
    before = spec_counters(base)
    start = threading.Event()
    results = [{} for _ in range(n)]
    threads = [threading.Thread(target=run_request,
                                args=(base, doc, max_tokens, start, out))
               for doc, out in zip(docs, results)]
    for th in threads:
        th.start()
    start.set()
    for th in threads:
        th.join()
    after = spec_counters(base)

    errors = [r["error"] for r in results if "error" in r]
    if errors or any(not r["times"] for r in results):
        return {"n": n, "errors": errors or ["a stream produced no tokens"]}
    t0 = min(r["t0"] for r in results)
    firsts = [r["times"][0] for r in results]
    w0, w1 = max(firsts), min(r["times"][-1] for r in results)
    window = w1 - w0
    agg = (sum(tokens_at(r, w1) - tokens_at(r, w0) for r in results) / window
           if window > 0 else None)
    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted_tokens"] - before["accepted_tokens"]
    drafted = after["draft_tokens"] - before["draft_tokens"]
    prompt = sum(r["prompt_tokens"] for r in results)
    return {
        "n": n,
        "prompt_tokens": [r["prompt_tokens"] for r in results],
        "output_tokens": [r["counts"][-1] for r in results],
        "finish": [r["finish"] for r in results],
        "ttft": [f - t0 for f in firsts],
        "prefill_tok_s": prompt / (w0 - t0),
        "stream_tok_s": [(r["counts"][-1] - r["counts"][0])
                         / (r["times"][-1] - r["times"][0]) for r in results],
        "window_s": window,
        "decode_tok_s": agg,
        "tokens_per_step": 1 + accepted / drafts if drafts else None,
        "draft_acceptance": accepted / drafted if drafted else None,
        "total_s": max(r["times"][-1] for r in results) - t0,
    }


def report(s):
    if "errors" in s:
        print(f"x{s['n']}: FAILED {s['errors']}", flush=True)
        return
    agg = s["decode_tok_s"]
    agg_text = (f"{agg:6.1f} tok/s total, {agg / s['n']:5.1f} per stream "
                f"over {s['window_s']:.1f}s" if agg else "no common window")
    print(f"x{s['n']}: prompt {min(s['prompt_tokens']):,}-"
          f"{max(s['prompt_tokens']):,} tok | TTFT "
          + "/".join(f"{t:.1f}" for t in s["ttft"])
          + f" s (prefill {s['prefill_tok_s']:,.0f} tok/s) | decode "
          f"{agg_text} | per-stream "
          + "/".join(f"{v:.0f}" for v in s["stream_tok_s"])
          + f" | MTP {s['tokens_per_step']:.2f} tok/step, "
          f"{s['draft_acceptance']:.0%} accepted | out "
          + "/".join(map(str, s["output_tokens"]))
          + f" ({','.join(sorted(set(s['finish'])))})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--depth", type=int, default=100 * 1024)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--src", default=str(
        Path(__file__).resolve().parents[2] / "vllm-windows" / "vllm"))
    p.add_argument("--json", help="also write the raw stats here")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    docs = Docs(args.url, args.src, args.depth)
    print(f"depth {args.depth:,} tokens, max_tokens {args.max_tokens}, "
          f"levels {levels}; building {1 + sum(levels)} prompts...", flush=True)
    warm = docs.next()
    batches = [[docs.next() for _ in range(n)] for n in levels]

    print("warmup (cold prefill, 32 output tokens):", flush=True)
    w = run_level(args.url, [warm], 1, 32)
    if "errors" in w:
        report(w)
        return
    print(f"  TTFT {w['ttft'][0]:.1f} s for {w['prompt_tokens'][0]:,} tokens",
          flush=True)
    stats = []
    for batch in batches:
        s = run_level(args.url, batch, len(batch), args.max_tokens)
        report(s)
        stats.append(s)
    if args.json:
        Path(args.json).write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
