"""Split generated thinking text from the answer when a chat request enables thinking.

With thinking on, the chat template ends the prompt with "<think>\\n", so the model writes its reasoning first
and closes it with "</think>". Clients (Pi, Hermes) expect that text in reasoning_content, as Halogen sends it,
and send it back as assistant reasoning_content, which the template renders inside <think> again.
With thinking off the splitter passes every character through unchanged.
"""

END = "</think>"


class ThinkSplitter:
    def __init__(self, enabled):
        self.enabled = enabled
        self.thinking = enabled
        self.pending = ""
        self.reasoning = ""
        self.started = False
        self.answer_started = False

    def feed(self, text):
        """Return (reasoning_delta, content_delta) for one piece of decoded text."""
        if not self.thinking:
            return "", self._answer(text)
        combined = self.pending + text
        end = combined.find(END)
        if end >= 0:
            self.thinking, self.pending = False, ""
            return self._reason(combined[:end]), self._answer(combined[end + len(END):])
        # Hold back a suffix that could start the end tag in the next piece.
        keep = next((length for length in range(min(len(combined), len(END) - 1), 0, -1)
                     if END.startswith(combined[-length:])), 0)
        self.pending = combined[len(combined) - keep:]
        return self._reason(combined[:len(combined) - keep]), ""

    def flush(self):
        """Text held at the end of generation: reasoning that never reached the end tag."""
        if not self.thinking:
            return "", ""
        text, self.pending = self.pending, ""
        return self._reason(text), ""

    def _reason(self, text):
        if not self.started:
            text = text.lstrip("\n")
            self.started = bool(text)
        self.reasoning += text
        return text

    def _answer(self, text):
        if self.enabled and not self.answer_started:
            # The template writes "</think>\n\n" before the answer and trims it on replay.
            # Strip newlines only, so an answer that starts with indented code keeps its indentation.
            text = text.lstrip("\n")
            self.answer_started = bool(text)
        return text
