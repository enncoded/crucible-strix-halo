"""The server command of a lease: serial mode must say --serial."""
import argparse
import unittest
from pathlib import Path
from bench.agents import lease


def arguments(mode):
    return argparse.Namespace(mode=mode, checkpoint=Path('ckpt'), cache_mib=64)


class ServerCommandTests(unittest.TestCase):
    def test_serial_mode_selects_the_serial_stack(self):
        command = lease.server_command(arguments('serial'), {'concurrency': 4}, Path('wrapper'))
        self.assertIn('--serial', command)
        self.assertNotIn('--streams', command)

    def test_batch_mode_names_the_streams(self):
        command = lease.server_command(arguments('batch'), {'concurrency': 4}, Path('wrapper'))
        self.assertNotIn('--serial', command)
        self.assertEqual(command[command.index('--streams') + 1], '4')


if __name__ == '__main__':
    unittest.main()
