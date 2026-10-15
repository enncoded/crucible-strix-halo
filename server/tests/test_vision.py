import base64
import io
import json
from unittest import mock
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from server.api import ApiError, validate_request
from server import vision
from server.vision import VisionSettings, expand_pads, smart_resize, IMAGE_PAD, VISION_START, VISION_END

try:
    from PIL import Image
except ImportError:  # Pillow is needed only for --vision.
    Image = None

CHECKPOINT = os.environ.get("CRUCIBLE_TEST_CKPT")
SETTINGS = VisionSettings()


def png_url(width=64, height=48, color=(200, 30, 60), fmt="PNG"):
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def chat(*content, role="user", extra=()):
    return dict(messages=list(extra) + [dict(role=role, content=list(content))], max_tokens=4)


def image_part(url):
    return dict(type="image_url", image_url=dict(url=url))


class ResizeTests(unittest.TestCase):
    def test_reference_engine_rule(self):
        self.assertEqual(smart_resize(1080, 1920), (1088, 1920))
        self.assertEqual(smart_resize(10, 10), (256, 256))
        self.assertEqual(smart_resize(2160, 3840), (1440, 2560))

    def test_aspect_and_zero(self):
        for size in ((0, 4), (1, 300)):
            with self.assertRaises(ApiError):
                smart_resize(*size)


class ExpandTests(unittest.TestCase):
    def images(self, *tokens):
        return [SimpleNamespace(tokens=n) for n in tokens]

    def test_expands_template_pads_in_order(self):
        ids = [1, VISION_START, IMAGE_PAD, VISION_END, 2, VISION_START, IMAGE_PAD, VISION_END]
        out, offsets = expand_pads(ids, self.images(3, 2))
        self.assertEqual(out, [1, VISION_START] + [IMAGE_PAD] * 3 + [VISION_END, 2, VISION_START] + [IMAGE_PAD] * 2 +
                         [VISION_END])
        self.assertEqual(offsets, [2, 8])

    def test_mention_stays_one_token(self):
        ids = [IMAGE_PAD, 5, VISION_START, IMAGE_PAD, VISION_END, IMAGE_PAD]
        out, offsets = expand_pads(ids, self.images(4))
        self.assertEqual(out, [IMAGE_PAD, 5, VISION_START] + [IMAGE_PAD] * 4 + [VISION_END, IMAGE_PAD])
        self.assertEqual(offsets, [3])

    def test_count_mismatch_is_400(self):
        with self.assertRaises(ApiError) as caught:
            expand_pads([VISION_START, IMAGE_PAD, VISION_END] * 2, self.images(1))
        self.assertEqual(caught.exception.status, 400)

    def test_no_images_is_identity(self):
        self.assertEqual(expand_pads([1, IMAGE_PAD, 2], []), ([1, IMAGE_PAD, 2], []))


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class RequestTests(unittest.TestCase):
    def check_error(self, body, text, vision_settings=SETTINGS):
        with self.assertRaises(ApiError) as caught:
            validate_request(body, "chat", "crucible", vision=vision_settings)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn(text, str(caught.exception))

    def test_vision_off_names_the_flag(self):
        self.check_error(chat(dict(type="text", text="What is this?"), image_part(png_url())), "--vision", None)

    def test_text_only_request_is_unchanged(self):
        body = chat(dict(type="text", text="a"), dict(type="text", text="b"))
        on = validate_request(body, "chat", "crucible", vision=SETTINGS)
        off = validate_request(body, "chat", "crucible")
        self.assertEqual(on.messages, off.messages)
        self.assertEqual(on.messages[0]["content"], "ab")
        self.assertEqual(on.images, ())

    def test_image_parts_decode_in_order(self):
        body = chat(dict(type="text", text="Compare"), image_part(png_url(64, 48)), image_part(png_url(300, 100, fmt="JPEG")))
        request = validate_request(body, "chat", "crucible", vision=SETTINGS)
        self.assertEqual(request.messages[0]["content"],
                         [dict(type="text", text="Compare"), dict(type="image"), dict(type="image")])
        self.assertEqual([(i.height, i.width) for i in request.images], [(224, 320), (160, 448)])
        self.assertEqual([i.tokens for i in request.images], [7 * 10, 5 * 14])
        self.assertEqual(len(request.images[0].rgb), 224 * 320 * 3)

    def test_digest_tracks_pixels(self):
        first = validate_request(chat(image_part(png_url(color=(1, 2, 3)))), "chat", "crucible", vision=SETTINGS)
        second = validate_request(chat(image_part(png_url(color=(1, 2, 4)))), "chat", "crucible", vision=SETTINGS)
        again = validate_request(chat(image_part(png_url(color=(1, 2, 3)))), "chat", "crucible", vision=SETTINGS)
        self.assertNotEqual(first.images[0].digest, second.images[0].digest)
        self.assertEqual(first.images[0].digest, again.images[0].digest)

    def test_tool_message_image_is_accepted(self):
        call = dict(id="call_1", type="function", function=dict(name="shot", arguments="{}"))
        tools = [dict(type="function", function=dict(name="shot", parameters=dict(type="object", properties={})))]
        body = dict(messages=[dict(role="user", content="Take a screenshot."),
                              dict(role="assistant", content=None, tool_calls=[call]),
                              dict(role="tool", tool_call_id="call_1", content=[image_part(png_url())])],
                    tools=tools, max_tokens=4)
        request = validate_request(body, "chat", "crucible", vision=SETTINGS)
        self.assertEqual(len(request.images), 1)

    def test_rejections(self):
        self.check_error(chat(image_part("https://example.com/a.png")), "does not fetch image URLs")
        self.check_error(chat(image_part("data:image/png,abc")), "base64")
        self.check_error(chat(image_part("data:image/png;base64,")), "empty")
        self.check_error(chat(image_part("data:image/png;base64," + base64.b64encode(b"not an image").decode())),
                         "could not be decoded.")
        self.check_error(chat(image_part(png_url(2000, 5))), "aspect ratio")
        self.check_error(chat(image_part(png_url()), role="assistant", extra=[dict(role="user", content="hi")]),
                         "user and tool messages")
        self.check_error(dict(messages=[dict(role="system", content=[image_part(png_url())]),
                                        dict(role="user", content="hi")], max_tokens=4), "user and tool messages")
        self.check_error(chat(*[image_part(png_url())] * 3), "at most 2 images", VisionSettings(max_images=2))
        self.check_error(chat(image_part(png_url(400, 400))), "limit before resizing", VisionSettings(max_pixels=10000))
        self.check_error(chat(image_part(png_url(220, 220))), "limit before resizing", VisionSettings(max_pixels=10000))
        self.check_error(chat(dict(type="video_url", video_url=dict(url="x")), image_part(png_url())),
                         "Only text message content")

    def test_engine_payload(self):
        request = validate_request(chat(image_part(png_url())), "chat", "crucible", vision=SETTINGS)
        [entry] = vision.engine_images(request.images, [7], ["/dev/shm/x/0a"])
        image = request.images[0]
        self.assertEqual(entry, dict(offset=7, height=image.height, width=image.width, digest=image.digest,
                                     path="/dev/shm/x/0a"))


class StubTokenizer:
    """One image placeholder between two text tokens."""
    def encode(self, request):
        return [1, VISION_START, IMAGE_PAD, VISION_END, 2]
    def encode_images(self, request):
        return expand_pads(self.encode(request), request.images)
    def cache_prefix_tokens(self, request, ids):
        return 0
    def stream(self):
        return None
    def step(self, stream, token):
        return ""
    def decode(self, ids):
        return ""


class StubBackend:
    """One token, then done. `submitted` (optional) records each job."""
    context = 4096

    def __init__(self, submitted=None):
        self.submitted = submitted

    def submit(self, ids, request, **options):
        import queue
        import threading
        if self.submitted is not None:
            self.submitted.append((ids, request, options))
        events = queue.Queue()
        events.put(dict(event="token", id=248046))
        events.put(dict(event="done", prompt_tokens=len(ids), completion_tokens=1, finish_reason="stop"))
        return SimpleNamespace(events=events, cancelled=threading.Event())


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class HttpVisionTests(unittest.TestCase):
    """The HTTP path: vision off refuses images; vision on sends offsets and the engine payload."""

    def post(self, body, vision):
        from email.message import Message
        from server.http import Service, handler_for

        submitted = []
        handler = handler_for(Service(StubBackend(submitted), StubTokenizer(), max_body_mib=8, vision=vision))
        instance = handler.__new__(handler)
        instance.path = "/v1/chat/completions"
        encoded = json.dumps(body).encode()
        instance.headers = Message()
        for key, value in {"Host": "127.0.0.1:8796", "Content-Length": str(len(encoded)),
                           "Content-Type": "application/json"}.items():
            instance.headers[key] = value
        instance.server = SimpleNamespace(server_address=("127.0.0.1", 8796))
        instance.rfile, instance.wfile = io.BytesIO(encoded), io.BytesIO()
        status = []
        instance.send_response = status.append
        instance.send_header = lambda name, value: None
        instance.end_headers = lambda: None
        instance.do_POST()
        return status[0], json.loads(instance.wfile.getvalue()), submitted

    def test_vision_off_is_400(self):
        status, body, submitted = self.post(chat(image_part(png_url())), None)
        self.assertEqual(status, 400)
        self.assertIn("--vision", body["error"]["message"])
        self.assertEqual(submitted, [])

    def test_vision_on_submits_offsets(self):
        status, body, submitted = self.post(chat(image_part(png_url())), SETTINGS)
        self.assertEqual(status, 200)
        [(ids, request, options)] = submitted
        self.assertEqual(options["image_offsets"], [2])
        self.assertEqual(ids.count(IMAGE_PAD), request.images[0].tokens)
        self.assertEqual(body["usage"]["prompt_tokens"], len(ids))

    def test_text_request_has_no_image_options(self):
        status, _, submitted = self.post(dict(messages=[dict(role="user", content="hi")], max_tokens=4), SETTINGS)
        self.assertEqual(status, 200)
        self.assertNotIn("image_offsets", submitted[0][2])


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class PixelLifetimeTests(unittest.TestCase):
    """after a request ends, nothing keeps its decoded pixels. Pi sends every earlier screenshot again on each
    turn, and the cyclic collector seldom runs, so a handler-to-response cycle grew the server RSS by each request."""

    def test_no_image_survives_the_request_without_the_cyclic_collector(self):
        import gc
        from server.http import Service, handler_for
        from server.vision import VisionImage

        def alive():
            return sum(isinstance(o, VisionImage) for o in gc.get_objects())

        handler = handler_for(Service(StubBackend(), StubTokenizer(), max_body_mib=8, vision=SETTINGS))
        body = json.dumps(chat(image_part(png_url(320, 320)))).encode()
        raw = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:8796\r\nContent-Type: application/json\r\n"
               b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        gc.disable()
        try:
            before = alive()
            instance = handler.__new__(handler)
            instance.rfile, instance.wfile = io.BytesIO(raw), io.BytesIO()
            instance.server = SimpleNamespace(server_address=("127.0.0.1", 8796))
            instance.client_address = ("127.0.0.1", 40000)
            instance.close_connection = True
            instance.handle_one_request()
            self.assertIn(b" 200 ", instance.wfile.getvalue().split(b"\r\n", 1)[0] + b" ")
            del instance
            self.assertEqual(alive(), before)
        finally:
            gc.enable()


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class ThinkCapImageTests(unittest.TestCase):
    def test_continuation_keeps_the_images(self):
        # The thinking cap prefills prompt + reasoning again on a second job; the image rows must still get pixels.
        from server.tests.serve_fixtures import invoke
        from server.tests.test_think_budget import ANSWER, IM_END, REASON, BudgetTokenizer, SequenceBackend

        class Tokenizer(BudgetTokenizer):
            def encode(self, request):
                return [10, VISION_START, IMAGE_PAD, VISION_END, 11]
            def encode_images(self, request):
                return expand_pads(self.encode(request), request.images)

        backend = SequenceBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", 9))
        body = chat(image_part(png_url())) | dict(max_tokens=2048, enable_thinking=True)
        response = invoke(backend, Tokenizer(), body, service_options=dict(vision=SETTINGS, max_body_mib=8))
        self.assertEqual(response.statuses, [200])
        self.assertEqual([options.get("image_offsets") for _, _, options in backend.submissions], [[2], [2]])
        first, second = (ids for ids, _, _ in backend.submissions)
        self.assertEqual(second[:len(first)], first)


@unittest.skipUnless(Image and CHECKPOINT, "Set CRUCIBLE_TEST_CKPT and install Pillow.")
class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.tokenization import ModelTokenizer
        cls.adapter = ModelTokenizer(CHECKPOINT)

    def test_expanded_prompt_and_cache_prefix(self):
        body = dict(messages=[dict(role="user", content=[dict(type="text", text="Describe"), image_part(png_url())]),
                              dict(role="assistant", content="A red box."),
                              dict(role="user", content="And the color code?")], max_tokens=4)
        request = validate_request(body, "chat", "crucible", vision=SETTINGS)
        ids, offsets = self.adapter.encode_images(request)
        n = request.images[0].tokens
        self.assertEqual(ids[offsets[0] - 1], VISION_START)
        self.assertEqual(ids[offsets[0]:offsets[0] + n], [IMAGE_PAD] * n)
        self.assertEqual(ids[offsets[0] + n], VISION_END)
        prefix = self.adapter.cache_prefix_tokens(request, ids)
        self.assertGreater(prefix, offsets[0] + n)

    def test_text_only_ids_unchanged(self):
        body = dict(messages=[dict(role="user", content="<|image_pad|> is a token.")], max_tokens=4)
        on = validate_request(body, "chat", "crucible", vision=SETTINGS)
        off = validate_request(body, "chat", "crucible")
        self.assertEqual(self.adapter.encode(on), self.adapter.encode(off))
        # RA-8: typed special-token text in content is text; the old rule ("parse") made it a real pad.
        self.assertNotIn(IMAGE_PAD, self.adapter.encode(on))
        from server.tokenization import ModelTokenizer
        self.assertIn(IMAGE_PAD, ModelTokenizer(CHECKPOINT, content_special_tokens="parse").encode(on))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class BackendPayloadTests(unittest.TestCase):
    """Text payloads stay byte-identical; image payloads carry the engine image list."""

    def test_payloads(self):
        import subprocess
        import sys
        import tempfile
        from unittest.mock import patch
        from server.backend import EngineBackend
        fake = Path(__file__).parent / "fake_engine.py"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "requests.jsonl"
            popen = subprocess.Popen
            env = dict(os.environ, FAKE_REQUEST_LOG=str(log), FAKE_CONTEXT="4096")
            with patch("server.backend.subprocess.Popen",
                       # The backend passes env= when it removes switches (CRUCIBLE_PI_AI_DIR in a Pi transport run).
                       side_effect=lambda *a, **k: popen([sys.executable, str(fake)], **(k | dict(env=env)))):
                backend = EngineBackend("unused", "unused", context=4096)
            try:
                text = validate_request(dict(messages=[dict(role="user", content="hi")], max_tokens=1), "chat",
                                        "crucible", vision=SETTINGS)
                image = validate_request(chat(image_part(png_url())), "chat", "crucible", vision=SETTINGS)
                for job in (backend.submit([1, 2], text), backend.submit([5] + [IMAGE_PAD] * image.images[0].tokens,
                                                                         image, image_offsets=[1])):
                    while job.events.get(timeout=5)["event"] != "done":
                        pass
            finally:
                backend.close()
            first, second = (json.loads(line) for line in log.read_text().splitlines())
        self.assertNotIn("images", first)
        self.assertEqual(second["images"][0]["offset"], 1)
        self.assertEqual(second["images"][0]["digest"], image.images[0].digest)
        self.assertNotIn("rgb", second["images"][0])


@unittest.skipUnless(Image, "Pillow is needed for image tests.")
class InputLimitTests(unittest.TestCase):
    """Release audit RA-2 to RA-5: limits that hold before a pixel is decoded."""

    def check_error(self, body, text, vision_settings=SETTINGS):
        with self.assertRaises(ApiError) as caught:
            validate_request(body, "chat", "crucible", vision=vision_settings)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn(text, str(caught.exception))

    def test_defaults_cover_the_graded_set_and_screenshots(self):
        # The context window, not the engine line, limits a long screenshot session.
        self.assertEqual(SETTINGS.max_images, 256)
        self.assertGreaterEqual(SETTINGS.max_image_bytes, 8 << 20)
        self.assertFalse(hasattr(SETTINGS, "max_line_bytes"))

    def test_image_tokens_must_fit_the_context_before_the_next_decode(self):
        # Review F2: without the old line cap, 256 full-size images were about 2.8 GB of RGB per request. The running
        # image-token total is checked from each header, before that image's pixels are decoded.
        one = (smart_resize(400, 400)[0] // 32) * (smart_resize(400, 400)[1] // 32)
        settings = VisionSettings(max_image_tokens=2 * one + 1)
        validate_request(chat(*[image_part(png_url(400, 400))] * 2), "chat", "crucible", vision=settings)
        loads, original = [], Image.Image.convert

        def counted(image, *args, **kwargs):
            loads.append(image.size)
            return original(image, *args, **kwargs)
        with mock.patch.object(Image.Image, "convert", counted):
            # Pi compacts its history when the text says "exceeds the context window".
            self.check_error(chat(*[image_part(png_url(400, 400))] * 3), "exceeds the context window", settings)
        self.assertEqual(len(loads), 2)

    def test_the_server_context_bounds_the_image_tokens(self):
        self.assertEqual(VisionSettings().max_image_tokens, 262144)

    def test_encoded_size_is_checked_before_base64_decode(self):
        url = png_url()
        with mock.patch.object(vision.base64, "b64decode", side_effect=AssertionError("decoded")):
            self.check_error(chat(image_part(url)), "encoded image", VisionSettings(max_image_bytes=16))

    def test_pillow_pixel_limit_follows_the_setting(self):
        vision.decode_bytes(base64.b64decode(png_url().split(",")[1]), VisionSettings(max_pixels=65536))
        self.assertEqual(Image.MAX_IMAGE_PIXELS, 65536 * vision.DECODE_FACTOR)

    def test_decoder_error_text_is_fixed(self):
        bad = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
        with self.assertRaises(ApiError) as caught:
            validate_request(chat(image_part(bad)), "chat", "crucible", vision=SETTINGS)
        self.assertEqual(str(caught.exception), "The image could not be decoded.")
        self.assertNotIn("0x", str(caught.exception))
