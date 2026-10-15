import importlib.util
import pathlib
import unittest

PATH=pathlib.Path(__file__).resolve().parents[1]/"bench/moe_shared_results.py"
spec=importlib.util.spec_from_file_location("moe_shared_results",PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    lines=["SHARED_FIRST C=2 PF=4 cache=default mode0=original mode1=shared R1=unchanged"]
    for work in ["R1","R8","B8"]:lines.append(f"EQUALITY {work} mode=1 H=1 logits=1 ids=1")
    for work in ["R8","B8"]:lines.append(f"SHARED_SAMPLES {work} "+" ".join(["10.000000/9.000000"]*32))
    for round in range(1,7):
        for mode in [0,1]:lines.append(f"SHARED_SPEC mode={mode} round={round} tok_s=100.000000 steps=30 decode_ms=1110.000000 ids_equal=1")
    for round in [1,2]:
        for mode in [0,1]:lines.append(f"SHARED_BATCH mode={mode} round={round} aggregate_tok_s=128.000000 wall_ms=62.500000 ids_equal=1")
    return "\n".join(lines)


class ResultsTests(unittest.TestCase):
    def test_complete(self):
        result=module.parse(fixture(),0)
        self.assertEqual(result["steps"]["R8"]["saved_ms"],1.)
        self.assertEqual(result["steps"]["R8"]["ci95_ms"],[1.,1.])
        self.assertEqual(result["spec"]["decode_saved_ms"],0.)

    def test_rejects_missing_duplicate_nonfinite_or_failed_job(self):
        for text,code in [(fixture(),1),(fixture().replace("EQUALITY R8 mode=1 H=1 logits=1 ids=1",""),0),
                          (fixture()+"\nSHARED_SAMPLES R8 10/9",0),
                          (fixture().replace("10.000000/9.000000","nan/9.000000",1),0),
                          (fixture().replace("SHARED_SPEC mode=1 round=6 tok_s=100.000000 steps=30 decode_ms=1110.000000 ids_equal=1",""),0)]:
            with self.subTest(code=code,text=text[-60:]):
                with self.assertRaises(ValueError):module.parse(text,code)


if __name__=="__main__":unittest.main()
