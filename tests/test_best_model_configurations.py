import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_best_model_configurations_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "best_model_configurations.py"
    spec = importlib.util.spec_from_file_location("best_model_configurations", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBestModelConfigurations(unittest.TestCase):
    def test_find_best_configuration_uses_lowest_mae_and_only_config_columns(self):
        best_model_configurations = load_best_model_configurations_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_dir = Path(tmp_dir) / "20260601-120000_db0_mlp_validation"
            result_dir.mkdir()
            pd.DataFrame(
                {
                    "Unnamed: 0": [0, 1],
                    "value": [0.5, 0.2],
                    "N": [3.0, 7.0],
                    "window_size_days_10": [4.0, 9.0],
                    "healthy_installation_id": [2, 2],
                    "prone_installation_id": [3, 3],
                    "mae_mean": [0.4, 0.1],
                    "mae_std": [0.01, 0.02],
                    "rmse_max": [0.8, 0.5],
                    "count_min": [20, 18],
                }
            ).to_csv(result_dir / "result.csv", index=False)

            row = best_model_configurations.find_best_configuration(result_dir / "result.csv")

        self.assertEqual(row["dataset"], "db0")
        self.assertEqual(row["model"], "mlp")
        self.assertEqual(row["config"], "N=7, window_size_days_10=9")
        self.assertEqual(row["score"], 0.1)

    def test_main_writes_summary_table(self):
        best_model_configurations = load_best_model_configurations_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cm_path = tmp_path / "cm"
            cm_path.mkdir()
            self._write_result(cm_path / "20260601-120000_db0_mlp", [0.3, 0.1], [2, 5])
            self._write_result(cm_path / "20260601-120002_db0_mlp", [0.05, 0.2], [9, 1])
            self._write_result(cm_path / "20260601-120001_db1_rf", [0.2, 0.4], [8, 1])

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    best_model_configurations.main()
            finally:
                os.chdir(original_cwd)

            result = pd.read_csv(cm_path / "best_model_configurations.csv")
            latex_table = (cm_path / "best_model_configurations.tex").read_text()

        self.assertEqual(list(result.columns), ["dataset", "model", "config"])
        self.assertEqual(result.loc[0].to_dict(), {"dataset": "db0", "model": "mlp", "config": "N=9"})
        self.assertEqual(result.loc[1].to_dict(), {"dataset": "db1", "model": "rf", "config": "N=8"})
        self.assertIn("Best model configurations - human readable", stdout.getvalue())
        self.assertIn("Best model configurations - LaTeX", stdout.getvalue())
        self.assertIn("N=9", stdout.getvalue())
        self.assertIn("\\begin{tabular}", latex_table)
        self.assertIn("dataset", latex_table)
        self.assertIn("model", latex_table)
        self.assertIn("N=9", latex_table)
        self.assertNotIn("...", latex_table)
        self.assertIn("\\begin{tabular}", stdout.getvalue())

    def _write_result(self, result_dir, mae_values, n_values):
        result_dir.mkdir()
        pd.DataFrame(
            {
                "mae_mean": mae_values,
                "mae_std": [0.01, 0.02],
                "N": n_values,
                "value": mae_values,
            }
        ).to_csv(result_dir / "result.csv", index=False)


if __name__ == "__main__":
    unittest.main()
