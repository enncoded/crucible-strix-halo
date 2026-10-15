"""the logit-scale rewrite of the MTP final mixer scales x exactly."""
import importlib.util
import unittest

HAVE = all(importlib.util.find_spec(m) for m in ("torch", "safetensors"))


@unittest.skipUnless(HAVE, "needs torch and safetensors")
class LogitScaleTest(unittest.TestCase):
    def test_mixer_output_scales(self):
        import torch
        from training.mtp_tune import export
        from training.mtp_tune.mtp_model import Hc

        class Ck:
            def __init__(self, t):
                self.tensors = t

            def t(self, name):
                return self.tensors[name]

        g = torch.Generator().manual_seed(0)
        pre = "mtp.hyper_connection_mixer."
        raw = {pre + "hc_norm.weight": torch.randn(10240, generator=g) * 0.1,
               pre + "input_mix_weight_down.weight": torch.randn(320, 10240, generator=g) * 0.01,
               pre + "input_mix_weight_up.weight": torch.randn(10240, 320, generator=g) * 0.05}
        ck = Ck(raw)
        scaled = export.scale_mixer({}, ck, 1.7)

        def mixer(norm, down):
            hc = Hc.__new__(Hc)
            torch.nn.Module.__init__(hc)
            hc.norm, hc.down, hc.up, hc.inject = norm, down, raw[pre + "input_mix_weight_up.weight"], None
            return hc

        H = torch.randn(3, 10240, generator=g, dtype=torch.float64)
        base, _ = mixer(raw[pre + "hc_norm.weight"].double(), raw[pre + "input_mix_weight_down.weight"].double())(H)
        new, _ = mixer(scaled["mixer.norm"].double(), scaled["mixer.down"].double())(H)
        torch.testing.assert_close(new, base * 1.7, rtol=1e-4, atol=1e-6)

    def test_rejects_exl3_parameter(self):
        from training.mtp_tune import export
        import torch
        with self.assertRaises(SystemExit):
            export.tensors(None, {"wq": torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()
