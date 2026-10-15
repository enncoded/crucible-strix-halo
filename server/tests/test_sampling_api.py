import math
import unittest
from unittest.mock import patch
from server.api import ApiError, validate_request

class SamplingApiTests(unittest.TestCase):
    def request(self, **options):
        return dict(prompt=[1, 2, 3], max_tokens=8) | options

    def test_all_sampling_fields(self):
        request = validate_request(self.request(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05,
                                               seed=18446744073709551615), "completion", "crucible")
        self.assertEqual(request.temperature, 0.7)
        self.assertEqual(request.top_k, 50)
        self.assertEqual(request.top_p, 0.9)
        self.assertEqual(request.min_p, 0.05)
        self.assertEqual(request.seed, 18446744073709551615)

    def test_greedy_default_is_unchanged(self):
        request = validate_request(self.request(), "completion", "crucible")
        self.assertEqual((request.temperature, request.top_k, request.top_p, request.min_p), (0, 0, 1, 0))
        self.assertIs(type(request.seed), int)
        self.assertTrue(0 <= request.seed < 1 << 64)

    def test_omitted_and_null_seeds_use_fresh_entropy(self):
        with patch("server.api.secrets.randbits", side_effect=[123, 456, 789]) as random:
            first = validate_request(self.request(temperature=0.7), "completion", "crucible")
            second = validate_request(self.request(temperature=0.7), "completion", "crucible")
            third = validate_request(self.request(temperature=0.7, seed=None), "completion", "crucible")
        self.assertEqual((first.seed, second.seed, third.seed), (123, 456, 789))
        self.assertEqual(random.call_count, 3)

    def test_explicit_zero_seed_does_not_use_entropy(self):
        with patch("server.api.secrets.randbits") as random:
            request = validate_request(self.request(seed=0), "completion", "crucible")
        self.assertEqual(request.seed, 0)
        random.assert_not_called()

    def test_invalid_sampling_fields(self):
        for options in (dict(temperature=True), dict(temperature=-1), dict(temperature=3),
                        dict(temperature=math.nan), dict(temperature=10 ** 1000),
                        dict(top_k=True), dict(top_k=-1), dict(top_k=248321), dict(top_p=1.1),
                        dict(top_p=False), dict(min_p=-0.1), dict(min_p=math.inf), dict(seed=-1),
                        dict(seed=True), dict(seed=1.5), dict(seed=18446744073709551616)):
            with self.subTest(options=options), self.assertRaises(ApiError):
                validate_request(self.request(**options), "completion", "crucible")

if __name__ == "__main__":
    unittest.main()
