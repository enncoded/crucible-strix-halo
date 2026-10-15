"""Render the checkpoint's real text chat template with its existing Jinja dependency."""
import json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from .api import ApiError

def raise_template_error(message):
    raise ApiError(str(message), "messages")

class ChatTemplate:
    def __init__(self, checkpoint):
        root = Path(checkpoint)
        config = json.loads((root / "tokenizer_config.json").read_text())
        path = root / "chat_template.jinja"
        template = path.read_text() if path.exists() else config.get("chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError("The checkpoint must contain a string chat template.")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = raise_template_error
        # HF's tojson filter preserves Unicode and does not HTML-escape the JSON.
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        self.template = environment.from_string(template)
        self.special_tokens = {key: value for key, value in config.items()
                               if key.endswith("_token") and isinstance(value, str)}

    def render(self, messages, enable_thinking=False, reasoning_effort="xhigh", tools=(), tool_choice="auto",
               add_generation_prompt=True, preserve_thinking=None):
        prepared = tuple(messages)
        instruction = None
        if tool_choice == "required":
            instruction = "Call one of the available tools. Do not give a text-only answer."
        elif isinstance(tool_choice, dict):
            instruction = "Call the function " + tool_choice["function"]["name"] + ". Do not call another function."
        if instruction:
            if prepared and prepared[0]["role"] == "system":
                prepared = (prepared[0] | dict(content=prepared[0]["content"] + "\n\n" + instruction),) + prepared[1:]
            else:
                prepared = (dict(role="system", content=instruction),) + prepared
        available = () if tool_choice == "none" else tools
        # The template keeps earlier reasoning unless preserve_thinking is false, so leave it undefined by default.
        preserve = {} if preserve_thinking is None else dict(preserve_thinking=preserve_thinking)
        return self.template.render(messages=list(prepared), tools=list(available), add_generation_prompt=add_generation_prompt,
                                    enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
                                    **preserve, **self.special_tokens)
