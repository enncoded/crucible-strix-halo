"""The server takes model metadata from the engine's ready event, not from its own constants."""
import unittest

from server.api import LEGACY_MODEL, STOP_IDS, ApiError, ModelInfo, validate_request
from server.backend import EngineBackend, ProtocolError
from server.batch_backend import BatchedEngineBackend
from server.http import Service

MODEL = dict(arch="qwen4_exp", vocab=248100, eos=STOP_IDS[1], layers=48, hidden=2560)


def ready_backend(cls=EngineBackend, **ready):
    backend = object.__new__(cls)
    backend.context = 64
    backend._validate_ready(dict(event="ready", context=64, cache_tail=True) | ready)
    return backend


class ReadyModel(unittest.TestCase):
    def test_ready_event_sets_the_model(self):
        backend = ready_backend(model=MODEL)
        self.assertEqual(backend.model, ModelInfo(**MODEL))

    def test_engine_before_issue_47_keeps_the_legacy_model(self):
        self.assertEqual(ready_backend().model, LEGACY_MODEL)
        self.assertEqual((LEGACY_MODEL.vocab, LEGACY_MODEL.eos), (248320, 248044))

    def test_invalid_model_fields_are_a_protocol_error(self):
        bad = [None, [], dict(MODEL, vocab=True), dict(MODEL, vocab=0), dict(MODEL, eos=248100),
               dict(MODEL, eos=7), dict(MODEL, layers="48"), dict(MODEL, arch=3), dict(MODEL, extra=1),
               {k: v for k, v in MODEL.items() if k != "hidden"}]
        for model in bad:
            with self.subTest(model=model), self.assertRaises(ProtocolError):
                ready_backend(model=model)

    def test_requests_use_the_engine_vocabulary(self):
        self.assertEqual(validate_request(dict(prompt=[248099], top_k=248100), "completion", "m", vocab=248100).prompt, (248099,))
        for body in (dict(prompt=[248100]), dict(prompt=[1], top_k=248101)):
            with self.subTest(body=body), self.assertRaises(ApiError):
                validate_request(body, "completion", "m", vocab=248100)

    def test_token_events_use_the_engine_vocabulary(self):
        service = Service(ready_backend(model=MODEL), tokenizer=None)
        request = validate_request(dict(prompt=[1], max_tokens=4), "completion", "crucible", vocab=248100)
        self.assertEqual(service.validate_event(dict(event="token", id=248099), [], 1, request), 248099)
        with self.assertRaises(ApiError):
            service.validate_event(dict(event="token", id=248100), [], 1, request)

    def test_batch_tokens_use_the_engine_vocabulary(self):
        from types import SimpleNamespace
        backend = object.__new__(BatchedEngineBackend)
        backend.model = ModelInfo(**MODEL)
        job = SimpleNamespace(payload=dict(max_tokens=4, stop_ids=[], ids=[1]))
        backend.active = {1: SimpleNamespace(job=job, count=1, generation=5)}
        token = lambda id: dict(event="token", request_id=1, slot_generation=5, id=id)
        backend._validate_event(token(248099), "spec_step", {1}, {})
        with self.assertRaises(ProtocolError):
            backend._validate_event(token(248100), "spec_step", {1}, {})


if __name__ == "__main__":
    unittest.main()
