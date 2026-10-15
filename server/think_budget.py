"""Close a thinking block that runs too long, as Halogen 0.15 does, so a capped turn still ends with an answer.

Halogen's rule (serve_api.py, 0.11.0 to 0.15.3):
- The answer room is max(1024, 15% of max_tokens) unless the operator sets a number (0 turns it off).
- The think block closes after max(1, max_tokens - room) generated tokens, or after the request's own
  thinking budget (max_thinking_tokens and its aliases) when that is smaller.
- The close is "\\n\\n" + Qwen's budget sentence + "\\n", then </think>, then "\\n\\n". It counts against max_tokens.
The count is on generated tokens, so the close lands on the same token with any decode mode or load.
"""
from dataclasses import dataclass

QWEN_BUDGET_SENTENCE = ("Considering the limited time by the user, I have to "
                        "give the solution based on the thinking directly now.")
THINK_END = "</think>"
ROOM_FLOOR = 1024
ROOM_PERCENT = 15


@dataclass(frozen=True)
class ThinkCap:
    tokens: int
    by: str


def answer_room(max_tokens, setting=None):
    """Tokens of max_tokens kept for the answer. None selects Halogen's policy; 0 is off."""
    if setting is None:
        return max(ROOM_FLOOR, max_tokens * ROOM_PERCENT // 100)
    return setting


def think_cap(max_tokens, budget=None, room_setting=None):
    """The generated-token count at which the server closes the think block, or None."""
    cap = ThinkCap(budget, "max_thinking_tokens") if budget else None
    room = answer_room(max_tokens, room_setting)
    if room > 0:
        limit = max(1, max_tokens - room)
        if cap is None or cap.tokens > limit:
            cap = ThinkCap(limit, "answer_room")
    return cap


def close_ids(tokenizer):
    """Token IDs of the forced close. The end tag is its own token, as the template writes it."""
    end = tokenizer.token_to_id(THINK_END)
    if end is None:
        raise ValueError("The tokenizer has no </think> token.")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    return tuple(encode("\n\n" + QWEN_BUDGET_SENTENCE + "\n")) + (end,) + tuple(encode("\n\n"))
