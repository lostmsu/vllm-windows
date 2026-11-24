"""Check qwen38-flash-next/chat_template.jinja against the model's stock template.

Renders through vLLM's own safe_apply_chat_template (including its
developer-to-system handling) with both templates:
- conversations the stock template accepts must render identically;
- system/developer messages after the first must appear as system turns at
  their original positions, with their content, and nothing else may change.

Run: .venv/Scripts/python.exe qwen38-flash-next/verify_chat_template.py
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

from vllm.renderers.hf import safe_apply_chat_template  # noqa: E402

HERE = Path(__file__).parent
MODEL = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
NEW = (HERE / "chat_template.jinja").read_text(encoding="utf-8")
ORIG = (HERE / "chat_template.orig.jinja").read_text(encoding="utf-8")
tok = AutoTokenizer.from_pretrained(MODEL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


def render(template, conv, tools=None, **kw):
    return safe_apply_chat_template(
        None,
        tok,
        [dict(m) for m in conv],
        tools=tools,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        **kw,
    )


def sys_(text):
    return {"role": "system", "content": text}


def dev(text):
    return {"role": "developer", "content": text}


def user(text):
    return {"role": "user", "content": text}


def asst(text, reasoning="thought", tool_calls=None):
    m = {"role": "assistant", "content": text, "reasoning_content": reasoning}
    if tool_calls:
        m["tool_calls"] = [
            {"type": "function", "id": f"c{i}", "function": {"name": "shell",
             "arguments": {"cmd": c}}}
            for i, c in enumerate(tool_calls)
        ]
    return m


def tool(text, i=0):
    return {"role": "tool", "content": text, "tool_call_id": f"c{i}"}


def turns(text):
    """[(role, body)] of the rendered <|im_start|>..<|im_end|> turns."""
    return re.findall(r"<\|im_start\|>(\w+)\n?(.*?)<\|im_end\|>", text, re.S)


failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f"\n     {detail}" if detail else ""))
    failures += not ok


# 1. Conversations the stock template accepts render byte-identically.
same_cases = {
    "system+user": ([sys_("S"), user("hi")], None, {}),
    "user only": ([user("hi")], None, {}),
    "multi-turn with reasoning": (
        [sys_("S"), user("a"), asst("b"), user("c")], None, {}),
    "tools, tool calls, two results": (
        [sys_("S"), user("ls"), asst("", tool_calls=["ls", "pwd"]),
         tool("x", 0), tool("y", 1), user("ok")], TOOLS, {}),
    "tools, no system": ([user("a"), asst("b"), user("c")], TOOLS, {}),
    "effort low": ([sys_("S"), user("a")], None, {"reasoning_effort": "low"}),
    "thinking off": ([sys_("S"), user("a")], TOOLS, {"enable_thinking": False}),
    "list content": (
        [{"role": "system", "content": [{"type": "text", "text": "S"}]},
         {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        None, {}),
}
for name, (conv, tools, kw) in same_cases.items():
    a, b = render(ORIG, conv, tools, **kw), render(NEW, conv, tools, **kw)
    check(f"unchanged: {name}", a == b)


# 2. Later system/developer messages: in place, as system turns, nothing else
# changes. Removing their turns must give the stock rendering without them.
def check_in_place(name, conv, tools=None, merge_free=True, **kw):
    out = render(NEW, conv, tools, **kw)
    expected_roles = []
    for i, m in enumerate(conv):
        role = m["role"]
        if role in ("system", "developer"):
            if i == 0:
                continue  # stays in the leading system block, as before
            expected_roles.append(("system", m["content"]))
        elif role == "tool":
            if not expected_roles or expected_roles[-1] != ("user", "<tool>"):
                expected_roles.append(("user", "<tool>"))
        else:
            expected_roles.append((role if role != "assistant" else role, None))
    got = turns(out)
    if got and got[0][0] == "system":
        # Leading block: system prompt, tools, reasoning effort. A later
        # system message can't come first: a user message must precede it.
        got = got[1:]
    got_roles = [
        ("user", "<tool>") if r == "user" and b.startswith("<tool_response>")
        else (r, b if r == "system" else None)
        for r, b in got
    ]
    ok = got_roles == expected_roles
    check(f"in place: {name}", ok,
          "" if ok else f"expected {expected_roles}\n     got      {got_roles}")
    if merge_free:
        rest = [m for i, m in enumerate(conv)
                if i == 0 or m["role"] not in ("system", "developer")]
        stripped = out
        for m in conv[1:]:
            if m["role"] in ("system", "developer"):
                block = f"<|im_start|>system\n{m['content']}<|im_end|>\n"
                stripped = stripped.replace(block, "", 1)
        check(f"  rest unchanged: {name}", stripped == render(ORIG, rest, tools, **kw))
    return out


check_in_place("between turns",
               [sys_("S"), user("a"), asst("b"), sys_("MID"), user("c")])
check_in_place("at the end",
               [sys_("S"), user("a"), asst("b"), user("c"), sys_("END")])
out = check_in_place("consecutive at start",
                     [sys_("S1"), sys_("S2"), user("a")])
check("  S1 stays in the leading block",
      out.startswith("<|im_start|>system\nReasoning effort is set to xhigh."
                     " Please think carefully through the task, validate key "
                     "assumptions, consider plausible alternatives, and "
                     "prioritize correctness, consistency, and clarity in "
                     "the final answer.\n\nS1<|im_end|>\n<|im_start|>system\nS2"))
check_in_place("consecutive mid",
               [user("a"), sys_("M1"), sys_("M2"), asst("b"), user("c")])
check_in_place("many, with tools",
               [sys_("S"), user("a"), sys_("M1"), asst("b"), sys_("M2"),
                sys_("M3"), user("c"), sys_("END")], TOOLS)
check_in_place("no leading system, tools", [user("a"), sys_("M"), asst("b"),
                                            user("c")], TOOLS)
check_in_place("between tool results",
               [sys_("S"), user("ls"), asst("", tool_calls=["ls", "pwd"]),
                tool("x", 0), sys_("M"), tool("y", 1)], TOOLS, merge_free=False)
check_in_place("after tool results",
               [sys_("S"), user("ls"), asst("", tool_calls=["ls"]),
                tool("x", 0), sys_("M")], TOOLS)
check_in_place("thinking off", [sys_("S"), user("a"), sys_("M")],
               enable_thinking=False)
check_in_place("multi-line content", [user("a"), sys_("line1\nline2")])
# Codex/Responses style: instructions as system, developer messages later.
# vLLM used to hoist all of these into one leading system message.
check_in_place("developer messages",
               [sys_("INSTR"), dev("D1"), user("a"), asst("b"), dev("D2"),
                user("c")], TOOLS)
check_in_place("leading developer", [dev("D1"), user("a"), dev("D2")])

out = render(NEW, [user("a"), {"role": "system",
                               "content": [{"type": "text", "text": "P1"},
                                           {"type": "text", "text": "P2"}]}])
check("list content mid-conversation", "<|im_start|>system\nP1P2<|im_end|>\n" in out)
out = render(NEW, [user("a"), sys_("  "), asst("b"), user("c")])
check("empty later system message is skipped (as at the start)",
      out == render(ORIG, [user("a"), asst("b"), user("c")]))

for name, conv in {
    "image in later system message": [user("a"), {"role": "system", "content": [
        {"type": "image_url", "image_url": {"url": "x"}}]}],
    "no user message": [sys_("S"), sys_("T")],
}.items():
    try:
        render(NEW, conv)
        check(f"still rejected: {name}", False)
    except Exception as e:
        check(f"still rejected: {name}", True, str(e))

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
