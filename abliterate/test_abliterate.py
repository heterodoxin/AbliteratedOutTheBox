import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("strata_abliterate", Path(__file__).with_name("abliterate.py"))
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)


class Calibration(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(81)
        self.comply = self.rng.normal(size=(5, 4, 12)).astype(np.float32)
        self.refuse = self.rng.normal(size=(8, 4, 12)).astype(np.float32) + 1
        self.clean = self.rng.normal(size=(4, 4, 12)).astype(np.float32)
        self.disclaimer = self.rng.normal(size=(4, 4, 12)).astype(np.float32) + 2

    def test_cpp_recipe(self):
        got = abl.directions(self.comply, self.refuse, self.clean, self.disclaimer)
        for layer in range(4):
            # Scalar-loop reference to the existing tools/abliterate/abliterate.cpp recipe.
            hc, hr = self.comply[:, layer].astype(float), self.refuse[:, layer].astype(float)
            mc, mr = hc.mean(axis=0), hr.mean(axis=0)
            direction = mr - mc
            for opening in hc:
                v = opening - mc
                v /= np.sqrt(sum(x * x for x in v)) + 1e-6
                direction -= sum(x * y for x, y in zip(direction, v)) * v
            direction /= np.sqrt(sum(x * x for x in direction)) + 1e-6
            cp = hc @ direction
            np.testing.assert_allclose(got[0][layer], direction, atol=3e-6)
            self.assertAlmostEqual(float(got[1][layer]), float(cp.mean() - 2 * cp.std()), places=5)
            self.assertAlmostEqual(float(got[4][layer]), float(cp.mean()), places=5)
            self.assertAlmostEqual(float(got[5][layer]), float((hr @ direction).mean()), places=5)

    def test_scale_free(self):
        base = abl.directions(self.comply, self.refuse, self.clean, self.disclaimer)
        large = abl.directions(self.comply * 100, self.refuse * 100, self.clean * 100, self.disclaimer * 100)
        np.testing.assert_allclose(large[0], base[0], atol=3e-6)
        np.testing.assert_allclose(large[1], base[1] * 100, atol=3e-4)

    def test_disclaimer_disabled(self):
        result = abl.directions(self.comply, self.refuse, disc_strength=-1)
        self.assertTrue((result[2] == 0).all())
        self.assertTrue((result[3] == 0).all())

    def test_nonfinite_rejected(self):
        self.comply[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            abl.directions(self.comply, self.refuse, disc_strength=-1)

    def test_gate_token_max_after_band_mean(self):
        r = np.array([[1., 0.], [1., 0.]])
        states = np.array([[[0., 0.], [2., 0.]], [[2., 0.], [0., 0.]]])
        self.assertEqual(abl.gate_score(states, r, np.zeros(2), np.ones(2)), 1.)


class FakeEngine(abl.Engine):
    def __init__(self, fire):
        self.layers, self.width, self.band, self.edit_first = 9, 10, [3, 4, 5, 6], 3
        self.controls, self.calls = [], []
        self.proc = type("Process", (), {"poll": lambda self: None})()
        self.fire = fire
        self.rng = np.random.default_rng(24)

    def control(self, command):
        self.controls.append(command)

    def generate(self, ids, count):
        self.calls.append((ids, count))
        return list(range(count))

    def capture(self, take=1):
        if take == 24:
            # Force a gate outcome independently of the random anchor geometry.
            return np.zeros((4, 24, 10), dtype=np.float32)
        return self.rng.normal(size=(4, take, 10)).astype(np.float32)


class RequestLifecycle(unittest.TestCase):
    def test_clean_request_has_no_direction_upload(self):
        engine = FakeEngine(False)
        output, _, fired = engine.answer(lambda s: [1], max_new=3, threshold=1e30, disc_strength=-1)
        self.assertFalse(fired)
        self.assertEqual(output, [0, 1, 2])
        self.assertFalse(any(c.startswith("SET") or c == "ON" for c in engine.controls))
        self.assertEqual(engine.controls[-1], "OFF")
        self.assertEqual(len(engine.calls), 15)  # 13 anchors, gate, answer

    def test_fired_request_uploads_then_edits_and_resets(self):
        engine = FakeEngine(True)
        _, _, fired = engine.answer(lambda s: [1], threshold=-1e30)
        self.assertTrue(fired)
        self.assertEqual(sum(c.startswith("SET") for c in engine.controls), 4)
        self.assertEqual(engine.controls[-2:], ["ON", "OFF"])
        self.assertEqual(len(engine.calls), 23)

    def test_probe_error_resets_adapter(self):
        engine = FakeEngine(False)
        engine.capture = lambda take=1: (_ for _ in ()).throw(RuntimeError("capture failed"))
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            engine.answer(lambda s: [1])
        self.assertEqual(engine.controls[-1], "OFF")

    def test_early_eos_is_forwarded_before_scoring(self):
        engine = FakeEngine(False)
        original = engine.generate

        def generate(ids, count):
            result = original(ids, count)
            return [42] if count == 25 else result

        engine.generate = generate
        engine.answer(lambda s: [1], threshold=1e30, disc_strength=-1)
        self.assertIn(([1, 42], 1), engine.calls)

    def test_nan_threshold_rejected_before_probes(self):
        engine = FakeEngine(False)
        with self.assertRaises(ValueError):
            engine.answer(lambda s: [1], threshold=float("nan"))
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":
    unittest.main()
