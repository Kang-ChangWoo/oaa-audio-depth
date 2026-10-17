"""Recipe preservation tests; no dataset or CUDA is needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_oaa import make_parser
from experiments.value_moments_20261017.run_pair import make_commands
from experiments.value_moments_20261017.probe import balanced_indices


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "baseline"
        self.run.mkdir()
        self.out = Path(self.tmp.name) / "results"
        args = vars(make_parser().parse_args(["--run-name", "source", "--nviews", "8",
                    "--batch-size", "3", "--accum", "11", "--epochs", "40",
                    "--warmup-ep", "8", "--stem-stride1", "--subset-aug"]))
        self.manifest = {"args": args, "data_module": "data_mp3d", "ddp_world": 1,
                         "stft": {"STFT_HOP": 16, "STFT_WIN": 400,
                                  "STFT_NFFT": 512, "STFT_WINDOW": 2823}}
        self.write_manifest()

    def write_manifest(self):
        (self.run / "manifest.json").write_text(json.dumps(self.manifest))

    def test_paired_commands_preserve_recipe(self):
        env, commands = make_commands(self.run, self.out, [0, 1, 2])
        self.assertEqual(len(commands), 6)
        self.assertEqual(env["STFT_HOP"], "16")
        self.assertEqual(env["STFT_WINDOW"], "2823")
        for index in (0, 2, 4):
            control, variant = (vars(make_parser().parse_args(cmd[2:]))
                                for _, cmd in commands[index:index+2])
            self.assertEqual(control["seed"], index // 2)
            self.assertEqual(control["epochs"], 40)
            self.assertEqual(control["accum"], 11)
            self.assertEqual(control["warmup_ep"], 8.)
            self.assertTrue(control["stem_stride1"])
            self.assertFalse(control["value_moments"])
            self.assertTrue(variant["value_moments"])
            changed = {k for k in control if control[k] != variant[k]}
            self.assertEqual(changed, {"run_name", "value_moments"})

    def test_missing_stft_or_unknown_architecture_fails(self):
        del self.manifest["stft"]["STFT_HOP"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing STFT_HOP"):
            make_commands(self.run, self.out, [0])
        self.manifest["stft"]["STFT_HOP"] = 16
        self.manifest["args"]["audio_backbone"] = "another_encoder"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unrecognised"):
            make_commands(self.run, self.out, [0])

    def test_missing_world_size_requires_explicit_provenance(self):
        del self.manifest["ddp_world"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "world size"):
            make_commands(self.run, self.out, [0])
        self.assertEqual(len(make_commands(self.run, self.out, [0], 1)[1]), 2)
        with self.assertRaisesRegex(ValueError, "single-GPU"):
            make_commands(self.run, self.out, [0], 2)

    def test_no_overwrite(self):
        _, commands = make_commands(self.run, self.out, [0])
        (self.out / commands[0][0]).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            make_commands(self.run, self.out, [0])

    def test_probe_subset_is_balanced_and_reproducible(self):
        samples = [(s, i) for s in ("a", "b", "c") for i in range(12)]
        a = balanced_indices(samples, 11, 1)
        self.assertEqual(a, balanced_indices(samples, 11, 1))
        self.assertEqual(len(set(a)), 11)
        counts = [sum(samples[i][0] == s for i in a) for s in ("a", "b", "c")]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(len(balanced_indices(samples, 100, 1)), len(samples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
