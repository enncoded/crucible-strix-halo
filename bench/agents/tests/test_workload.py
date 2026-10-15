import hashlib
import unittest
from bench.agents import workload


def text_ids(text):
    return [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 248320 for word in text.split()]


def request_ids(body):
    return [100] + text_ids(str(body['tools'])) + [token for message in body['messages']
            for token in [101, *text_ids(message.get('content', '')), 102]] + [103]


class WorkloadTests(unittest.TestCase):
    def build(self, session=0, seed=7):
        return workload.session(request_ids, text_ids, session=session, seed=seed,
                                context_tokens=2048, prefix_tokens=256, suffix_sizes=(64, 256), output_tokens=200)

    def test_public_fixed_growing_histories(self):
        turns = self.build()
        self.assertGreater(len(turns), 2)
        self.assertEqual(turns, self.build())
        counts = [turn['prompt_tokens'] for turn in turns]
        self.assertEqual(counts, sorted(counts))
        self.assertTrue(all(count + 200 <= 2048 for count in counts))
        # The next scripted answer and tool framing must also fit before a new turn can start.
        self.assertGreater(counts[-1] + 200, 2048 - 360)
        self.assertTrue(all(turn['body']['return_token_ids'] and turn['body']['stream'] for turn in turns))
        self.assertEqual({turn['suffix_target_tokens'] for turn in turns}, {64, 256})
        self.assertEqual(turns[0]['fixture_label'], 'new_shared_prefix_session')
        self.assertTrue(all(turn['teacher_forced'] for turn in turns))

    def test_shared_prefix_and_session_isolation(self):
        left, right = self.build(0), self.build(1)
        self.assertEqual(left[0]['body']['messages'][0], right[0]['body']['messages'][0])
        self.assertNotEqual(left[0]['request_sha256'], right[0]['request_sha256'])
        self.assertGreater(left[1]['shared_previous_tokens'], 0)

    def test_bounds_and_explicit_telemetry(self):
        for changes in [{'context_tokens': 100}, {'output_tokens': 199}, {'seed': -1}, {'suffix_sizes': (0,)}]:
            args = dict(context_tokens=2048, prefix_tokens=256, output_tokens=200, seed=7)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                workload.session(request_ids, text_ids, **{**args, **changes})
        self.assertTrue(self.build()[0]['body']['crucible_metrics'])


if __name__ == '__main__': unittest.main()
