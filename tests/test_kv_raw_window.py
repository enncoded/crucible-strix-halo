"""Address-lifetime tests only. GPU pooling and model gates remain separate."""
import struct
import unittest


def row_bits(position):
    return struct.pack("<128I", *(0x3f000000 + position * 131 + dimension for dimension in range(128)))


def full_block(block):
    return tuple(row_bits(block * 4 + token) for token in range(4))


def local_span(position, rows):
    first = position // 4 * 4
    return first, tuple(row_bits(p) for p in range(first, position + rows))


def local_block(block, first, span):
    return tuple(span[block * 4 + token - first] for token in range(4))


def ring_prepare(position, rows, slots=16):
    first = max(0, position - slots)
    ring = {p % slots: (p, row_bits(p)) for p in range(first, position + rows)}
    return ring


def ring_block(block, ring, slots=16):
    positions = tuple(block * 4 + token for token in range(4))
    if any(ring[p % slots][0] != p for p in positions):
        raise ValueError("ring overwrote an input before pooling")
    return tuple(ring[p % slots][1] for p in positions)


class RawWindowTests(unittest.TestCase):
    def test_local_prefill_span_bits(self):
        for position in (0, 1, 2, 3, 14, 15, 16, 127, 128, 2047, 65533, 131071, 262141):
            for rows in (1, 2, 3, 8, 128, 1024, 2048):
                first, span = local_span(position, rows)
                self.assertLessEqual(len(span), rows + 3)
                for block in range(position // 4, (position + rows) // 4):
                    self.assertEqual(local_block(block, first, span), full_block(block))

    def test_decode_ring_and_every_accepted_prefix(self):
        for position in range(128):
            for rows in range(1, 9):
                ring = ring_prepare(position, rows)
                for block in range(position // 4, (position + rows) // 4):
                    self.assertEqual(ring_block(block, ring), full_block(block))
                for accepted in range(rows + 1):
                    next_position = position + accepted
                    next_ring = dict(ring)
                    for p in range(next_position, next_position + 8):
                        next_ring[p % 16] = (p, row_bits(p))
                    for block in range(next_position // 4, (next_position + 8) // 4):
                        self.assertEqual(ring_block(block, next_ring), full_block(block))

    def test_naive_small_prefill_ring_fails(self):
        ring = ring_prepare(15, 128)
        with self.assertRaises(ValueError):
            ring_block(15 // 4, ring)


if __name__ == "__main__":
    unittest.main()
