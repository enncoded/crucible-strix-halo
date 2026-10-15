import json
import unittest
from unittest.mock import patch

import runtime_environment as policy


class EnvironmentTests(unittest.TestCase):
    def test_exact_central_allowlist(self):
        env = dict.fromkeys(policy._shared.RUNTIME_KNOBS, 'example')
        state = policy.snapshot(env)
        self.assertEqual(state['environment'], policy._shared.runtime_environment(env))
        self.assertEqual(state['unknown_crucible_keys'], [])
        policy.require_known(state)

    def test_unrelated_secret_is_not_collected(self):
        state = policy.snapshot({'OPENAI_API_KEY': 'fake-private-value'})
        self.assertNotIn('OPENAI_API_KEY', state['environment'])
        self.assertNotIn('fake-private-value', json.dumps(state))

    def test_unknown_names_are_sorted_and_values_are_redacted(self):
        state = policy.snapshot({'CRUCIBLE_ZZZ': 'never-show-z', 'CRUCIBLE_AAA': 'never-show-a'})
        self.assertEqual(state['unknown_crucible_keys'], ['CRUCIBLE_AAA', 'CRUCIBLE_ZZZ'])
        for name in state['unknown_crucible_keys']:
            self.assertEqual(state['environment'][name], '[REDACTED]')
        with self.assertRaises(ValueError) as stopped:
            policy.require_known(state)
        self.assertNotIn('never-show', str(stopped.exception))
        self.assertNotIn('never-show', json.dumps(state))

    def test_changed_policy_cannot_use_cached_allowlist(self):
        with patch.object(policy, 'policy_hashes', return_value={'changed': 'policy'}), \
                self.assertRaisesRegex(ValueError, 'changed after import'):
            policy.snapshot({})


if __name__ == '__main__': unittest.main()
