import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_best_score_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "best_score.py"
    spec = importlib.util.spec_from_file_location("best_score", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBestScoreMain(unittest.TestCase):
    def test_main_prints_rows_ordered_by_mae_mean(self):
        best_score = load_best_score_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cm_path = tmp_path / "cm"
            cm_path.mkdir()

            self._write_result(cm_path / "result_model_b", [0.4, 0.2])
            self._write_result(cm_path / "result_model_a", [0.3, 0.1])

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    best_score.main()
            finally:
                os.chdir(original_cwd)

        output = stdout.getvalue()
        self.assertIn("Best scores for column", output)
        self.assertLess(output.index("model_a"), output.index("model_b"))

    def _write_result(self, result_dir, mae_values):
        result_dir.mkdir()
        pd.DataFrame(
            {
                "mae_mean": mae_values,
                "other_mean": [value * 10 for value in mae_values],
                "ignored": [100, 200],
            }
        ).to_csv(result_dir / "result.csv", index=False)


if __name__ == "__main__":
    unittest.main()
