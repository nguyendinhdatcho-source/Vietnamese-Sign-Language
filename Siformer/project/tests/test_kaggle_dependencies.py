"""Safety checks for Kaggle's in-place dependency installer (no pip is run)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import kaggle_dependencies as helper


class KaggleDependencyTests(unittest.TestCase):
    def test_protected_runtime_does_not_install_on_failed_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("tensorflow>=2.18,<2.22\nmediapipe==0.10.21\n")
            with (patch.object(sys, "argv", ["kaggle_dependencies.py", "--requirements", str(requirements)]),
                  patch.object(helper, "check_runtime", side_effect=RuntimeError("incompatible runtime")),
                  patch.object(helper.subprocess, "run") as run):
                with self.assertRaisesRegex(RuntimeError, "incompatible runtime"):
                    helper.main()
                run.assert_not_called()

    def test_only_unmet_extras_are_passed_to_pip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            requirements = root / "requirements.txt"
            report = root / "environment_versions.json"
            requirements.write_text("tensorflow>=2.18,<2.22\nnumpy>=1.26,<2\n"
                                    "mediapipe==0.10.21\npackaging>=23,<27\n")
            before = {"tensorflow": "2.18.1", "keras": "3.8.0", "numpy": "1.26.4",
                      "nvidia-cudnn-cu12": "9.3.0", "packaging": "24.2"}
            after = dict(before, mediapipe="0.10.21")
            with (patch.object(sys, "argv", ["kaggle_dependencies.py", "--requirements", str(requirements),
                                             "--report", str(report)]),
                  patch.object(helper, "check_runtime", return_value=before),
                  patch.object(helper, "installed_versions", return_value=after),
                  patch.object(helper.md, "distributions", return_value=[]),
                  patch.object(helper.subprocess, "run") as run):
                helper.main()
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
            self.assertIn("mediapipe==0.10.21", command)
            self.assertFalse(any(item.startswith("tensorflow==") or item.startswith("numpy==") for item in command))
            constraints = report.with_name("kaggle_protected_constraints.txt").read_text()
            self.assertIn("tensorflow==2.18.1", constraints)
            self.assertIn("keras==3.8.0", constraints)
            self.assertIn("numpy==1.26.4", constraints)
            self.assertIn("nvidia-cudnn-cu12==9.3.0", constraints)
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads(report.read_text()), {})

    def test_compatible_runtime_skips_pip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            requirements = root / "requirements.txt"
            requirements.write_text("tensorflow>=2.18,<2.22\nnumpy>=1.26,<2\nmediapipe==0.10.21\n")
            versions = {"tensorflow": "2.18.1", "keras": "3.8.0", "numpy": "1.26.4",
                        "mediapipe": "0.10.21"}
            with (patch.object(sys, "argv", ["kaggle_dependencies.py", "--requirements", str(requirements),
                                             "--report", str(root / "environment_versions.json")]),
                  patch.object(helper, "check_runtime", return_value=versions),
                  patch.object(helper, "installed_versions", return_value=versions),
                  patch.object(helper.md, "distributions", return_value=[]),
                  patch.object(helper.subprocess, "run") as run):
                helper.main()
            self.assertEqual(run.call_count, 1)
            self.assertNotIn("pip", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
