import importlib.util
import pathlib
import unittest

SPEC = importlib.util.spec_from_file_location(
    "batch_union", pathlib.Path(__file__).resolve().parents[1] / "tools/batch_union.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnionTests(unittest.TestCase):
    def test_identical_routes(self):
        curve = MODULE.union_curve(list(range(10)) * 8, (8, 1, 1, 10))
        self.assertEqual([row["mean"] for row in curve], [10] * 8)
        self.assertEqual(curve[1]["subsets"], 28)

    def test_disjoint_routes(self):
        curve = MODULE.union_curve(list(range(80)), (8, 1, 1, 10))
        self.assertEqual([row["mean"] for row in curve], list(range(10, 81, 10)))

    def test_partial_overlap(self):
        routes = [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2, 5]
        curve = MODULE.union_curve(routes, (3, 1, 1, 4))
        self.assertEqual([row["mean"] for row in curve], [4, 5, 6])

    def test_bad_payload(self):
        for routes in ([1] * 10, list(range(9)), list(range(9)) + [512]):
            with self.assertRaises(ValueError):
                MODULE.union_curve(routes, (1, 1, 1, 10))


if __name__ == "__main__":
    unittest.main()
