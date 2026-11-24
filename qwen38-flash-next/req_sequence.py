"""Send a fixed sequence of chat requests and report each outcome and speed.

Usage: python req_sequence.py [base_url]
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
SHORT = "A train leaves at 14:35 and arrives at 17:10. How long is the trip? Answer briefly."
CODE = ("Write a Python function that parses ISO 8601 durations like P3DT4H5M6S "
        "into seconds, with docstring and a few doctests. No explanation outside the code.")

SEQUENCE = [
    ("short, thinking", SHORT, True, 1024),
    ("short, thinking (repeat)", SHORT, True, 1024),
    ("code, no thinking", CODE, False, 2000),
    ("code, thinking", CODE, True, 4000),
]


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


for name, prompt, thinking, max_tokens in SEQUENCE:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.6,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.perf_counter()
    try:
        r = post("/v1/chat/completions", body)
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"{name}: FAILED after {time.perf_counter() - t0:.1f}s: {e}")
        break
    dt = time.perf_counter() - t0
    n = r["usage"]["completion_tokens"]
    content = (r["choices"][0]["message"]["content"] or "").strip()
    print(f"{name}: {n} tokens in {dt:.1f}s = {n / dt:.0f} tok/s, "
          f"finish={r['choices'][0]['finish_reason']}, answer={content[:80]!r}")
