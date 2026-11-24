"""Live check: system/developer messages after the first are accepted (HTTP 200)
by /v1/chat/completions and /v1/responses, and a late one is followed.

Run: .venv/Scripts/python.exe qwen38-flash-next/req_system_positions.py [--url URL] [--key KEY]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
p.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-FP4-Local")
args = p.parse_args()


def post(path, body):
    req = urllib.request.Request(
        args.url + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {args.key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


S = lambda t: {"role": "system", "content": t}  # noqa: E731
U = lambda t: {"role": "user", "content": t}  # noqa: E731
A = lambda t: {"role": "assistant", "content": t}  # noqa: E731
TOOLS = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers",
    "parameters": {"type": "object", "properties": {
        "a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"]}}}]
CALLS = {"role": "assistant", "content": "", "tool_calls": [
    {"id": f"call_{i}", "type": "function",
     "function": {"name": "add", "arguments": json.dumps({"a": i, "b": 1})}}
    for i in (1, 2)]}


def T(i, out):
    return {"role": "tool", "tool_call_id": f"call_{i}", "content": out}


chat_cases = {
    "system between turns": [S("Be terse."), U("Hi"), A("Hello."),
                             S("Answer in French."), U("What is 2+2?")],
    "system at the end": [U("Hi"), A("Hello."), U("What is 2+2?"),
                          S("Answer with digits only.")],
    "consecutive systems, start and mid": [S("Be terse."), S("No emojis."),
                                           U("Hi"), A("Hello."),
                                           S("Answer in German."),
                                           S("Use one word."), U("Thanks")],
    "system between tool results": [S("Use the add tool."), U("1+1 and 2+1?"),
                                    CALLS, T(1, "2"), S("Be terse."), T(2, "3")],
}
failures = 0
for name, messages in chat_cases.items():
    body = {"model": args.model, "messages": messages, "max_tokens": 512,
            "reasoning_effort": "low"}
    if "tool" in name:
        body["tools"] = TOOLS
    status, resp = post("/v1/chat/completions", body)
    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
    print(f"{status} chat/completions  {name}: {text!r:.80}"
          if status == 200 else f"{status} chat/completions  {name}: {resp}")
    failures += status != 200

status, resp = post("/v1/responses", {
    "model": args.model, "instructions": "Be terse.",
    "input": [U("Hi"), A("Hello."), {"role": "developer",
                                     "content": "Answer in Spanish."},
              U("What is 2+2?"), S("Digits are fine.")],
    "max_output_tokens": 512, "reasoning": {"effort": "low"}})
text = next((c["text"] for o in resp.get("output", []) if o["type"] == "message"
             for c in o["content"]), None)
print(f"{status} responses         developer + system mid-input: {text!r:.80}"
      if status == 200 else f"{status} responses: {resp}")
failures += status != 200

# The late system message must take effect, not just be accepted.
status, resp = post("/v1/chat/completions", {
    "model": args.model, "max_tokens": 512, "reasoning_effort": "low",
    "messages": [U("Name a fruit."), A("Apple."), U("Another one?"),
                 S("Whatever the user asks, reply with exactly: PINEAPPLE")]})
text = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
ok = status == 200 and "PINEAPPLE" in text
print(f"{status} late system instruction followed: {ok} ({text!r:.60})")
failures += not ok

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
