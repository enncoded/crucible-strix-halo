#!/usr/bin/env python3
"""G15 CPU check: does a chat client's next prompt repeat the generated tokens exactly?

Session replay hits only when the new prompt starts with the previous prompt plus the generated tokens. A chat
client sends messages, and the template re-renders the assistant turn. This script renders typical assistant
turns (plain, thinking, tool call) and reports where the re-rendered prompt first differs. No GPU.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.template import ChatTemplate
from tokenizers import Tokenizer

IM_END = 248046
TOOL = dict(type="function", function=dict(name="read_file", description="Read a file.",
            parameters=dict(type="object", properties=dict(path=dict(type="string")), required=["path"])))


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    checkpoint = sys.argv[1]
    template, tokenizer = ChatTemplate(checkpoint), Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    history = [dict(role="system", content="You are a coding agent."), dict(role="user", content="Fix the bug.")]
    call = dict(type="function", function=dict(name="read_file", arguments=dict(path="engine/serve/serve.h")))
    tool_text = ('<tool_call>\n<function=read_file>\n<parameter=path>\nengine/serve.h\n</parameter>\n'
                 '</function>\n</tool_call>')
    report = {}
    # The generation prompt ends with "<think>\n" (thinking) or an empty think block; the model writes the rest.
    for thinking in (False, True):
        for name, content, calls in (("plain", "The bug is in Find.", None), ("tool_call", "", [call])):
            for follow_role in ("user", "tool"):
                if follow_role == "tool" and not calls: continue
                reasoning = "Check Find first." if thinking else ""
                generated = (reasoning + "\n</think>\n\n" if thinking else "") + (content or tool_text)
                message = dict(role="assistant", content=content) | (dict(tool_calls=calls) if calls else {})
                if thinking: message["reasoning_content"] = reasoning
                prompt = encode(template.render(history, thinking, tools=[TOOL]))
                expected = prompt + encode(generated) + [IM_END]
                follow = dict(role=follow_role, content="file text" if follow_role == "tool" else "Continue.")
                again = encode(template.render(history + [message, follow], thinking, tools=[TOOL]))
                at = first_difference(expected, again)
                report[f"{name}/thinking={int(thinking)}/next={follow_role}"] = dict(
                    hit=at == len(expected), matched=at, expected=len(expected), prompt=len(prompt),
                    expected_text=tokenizer.decode(expected[at:at + 12]),
                    rendered_text=tokenizer.decode(again[at:at + 12]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
