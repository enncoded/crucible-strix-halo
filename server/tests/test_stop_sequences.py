import random
import unittest
from server.api import ApiError
from server.stop_sequences import StopSequences, validate_stop

class StopSequenceTests(unittest.TestCase):
    def collect(self, stops, chunks):
        matcher = StopSequences(stops)
        result = "".join(matcher.feed(chunk) for chunk in chunks) + matcher.flush()
        return result, matcher.stopped

    def test_all_splits_and_unicode(self):
        raw = "Before 雪 STOP hidden"
        for split in range(len(raw) + 1):
            self.assertEqual(self.collect(("STOP",), [raw[:split], raw[split:]]), ("Before 雪 ", True))

    def test_overlap_uses_first_completed_stop(self):
        for split in range(4):
            self.assertEqual(self.collect(("abc", "b"), ["abc"[:split], "abc"[split:]]), ("a", True))
        self.assertEqual(self.collect(("abc", "bc"), ["abc"]), ("", True))

    def test_repeated_prefix_is_bounded(self):
        stop = "a" * 4095 + "b"
        matcher = StopSequences((stop,))
        output = matcher.feed("a" * 100_000)
        self.assertEqual(len(matcher.pending), 4095)
        self.assertEqual(len(output), 100_000 - 4095)
        self.assertEqual(matcher.feed("bhidden"), "")
        self.assertTrue(matcher.stopped)

    def test_random_chunking_matches_character_reference(self):
        rng = random.Random(97)
        for _ in range(200):
            stops = tuple("".join(rng.choices("ab雪", k=rng.randrange(1, 8))) for _ in range(4))
            raw = "".join(rng.choices("ab雪", k=80))
            expected = (raw, False)
            for index in range(1, len(raw) + 1):
                matches = [index - len(stop) for stop in stops if raw[:index].endswith(stop)]
                if matches:
                    expected = (raw[:min(matches)], True)
                    break
            split = rng.randrange(len(raw) + 1)
            self.assertEqual(self.collect(stops, [raw[:split], raw[split:]]), expected)

    def test_validation_bounds(self):
        for value in ("", [""], 1, [False], ["x"] * 5, "雪" * 1366, "\ud800"):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ApiError) as error:
                validate_stop(value)
            self.assertEqual(error.exception.param, "stop")
        self.assertEqual(validate_stop(None), ())
        self.assertEqual(validate_stop([]), ())
        self.assertEqual(validate_stop("END"), ("END",))

if __name__ == "__main__":
    unittest.main()
