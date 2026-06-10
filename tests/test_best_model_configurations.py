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
            pd.DataFrame(
                {
                    "dataset": ["db0", "db1"],
                    "model": ["mlp", "rf"],
                    "source": ["first", "second"],
                }
            ).to_csv(cm_path / "dataset_model_configuration.csv", index=False)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    best_model_configurations.main()
            finally:
                os.chdir(original_cwd)

            result = pd.read_csv(cm_path / "best_model_configurations_generated.csv")
            latex_table = (cm_path / "best_model_configurations_generated.tex").read_text()

        self.assertEqual(list(result.columns), ["Dataset", "Model", "Config", "Source"])
        self.assertEqual(
            result.loc[0].to_dict(),
            {"Dataset": "db0", "Model": "mlp", "Config": "N=9", "Source": "first"},
        )
        self.assertEqual(
            result.loc[1].to_dict(),
            {"Dataset": "db1", "Model": "rf", "Config": "N=8", "Source": "second"},
        )
        self.assertIn("Best model configurations - human readable", stdout.getvalue())
        self.assertIn("Best model configurations - LaTeX", stdout.getvalue())
        self.assertIn("N=9", stdout.getvalue())
        self.assertIn("\\begin{tabular}", latex_table)
        self.assertIn("Dataset", latex_table)
        self.assertIn("Model", latex_table)
        self.assertIn("N=9", latex_table)
        self.assertIn("Source", latex_table)
        self.assertIn("PV1", latex_table)
        self.assertIn("$MLP_1$", latex_table)
        self.assertNotIn("...", latex_table)
        self.assertIn("\\begin{tabular}", stdout.getvalue())

    def test_main_does_not_overwrite_existing_external_configuration(self):
        best_model_configurations = load_best_model_configurations_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cm_path = tmp_path / "cm"
            cm_path.mkdir()
            self._write_result(cm_path / "20260601-120000_db0_mlp", [0.3, 0.1], [2, 5])
            external_configuration = pd.DataFrame(
                {
                    "dataset": ["db0"],
                    "model": ["cnn"],
                    "config": ["complexity2^x=4, n=4, n_step=7"],
                }
            )
            external_configuration.to_csv(cm_path / "best_model_configurations.csv", index=False)

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    best_model_configurations.main()
            finally:
                os.chdir(original_cwd)

            external_after = pd.read_csv(cm_path / "best_model_configurations.csv")
            generated = pd.read_csv(cm_path / "best_model_configurations_generated.csv")

        self.assertEqual(external_after.to_dict("records"), external_configuration.to_dict("records"))
        self.assertIn("cnn", set(generated["Model"]))
        self.assertIn("mlp", set(generated["Model"]))

    def test_display_table_formats_names_wraps_configs_and_rounds_floats(self):
        best_model_configurations = load_best_model_configurations_module()
        table = pd.DataFrame(
            {
                "config": [
                    (
                        "Bins=37, window_size_days_10=22, "
                        "conv2D_shape_factor=0.0037580323666489, "
                        "regressor_degrees=15, bandwidth=0.0104285791052169"
                    )
                ]
            },
            index=pd.MultiIndex.from_tuples(
                [("db0", "complex_mlp")],
                names=["dataset", "model"],
            ),
        )

        human_readable = best_model_configurations.format_table_for_display(
            table,
            best_model_configurations.format_config_for_display,
        )
        latex_table = best_model_configurations.format_table_for_display(
            table,
            best_model_configurations.format_config_for_latex,
        )

        self.assertEqual(human_readable.index[0], ("PV1", "$MLP_2$"))
        self.assertIn("conv2D_shape_factor=0.004", human_readable.loc[("PV1", "$MLP_2$"), "Config"])
        self.assertIn("\nregressor_degrees=15", human_readable.loc[("PV1", "$MLP_2$"), "Config"])
        self.assertIn(r"\makecell[l]{", latex_table.loc[("PV1", "$MLP_2$"), "Config"])
        self.assertIn(r"conv2D\_shape\_factor=0.004", latex_table.loc[("PV1", "$MLP_2$"), "Config"])

    def test_sort_configurations_by_score_orders_models_within_dataset(self):
        best_model_configurations = load_best_model_configurations_module()
        table = pd.DataFrame(
            {
                "config": ["slow", "external", "fast", "other"],
                "score": [0.2, pd.NA, 0.1, 0.05],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("db0", "mlp"),
                    ("db0", "cnn"),
                    ("db0", "rf"),
                    ("db1", "lr"),
                ],
                names=["dataset", "model"],
            ),
        )

        sorted_table = best_model_configurations.sort_configurations_by_score(table)

        self.assertEqual(
            list(sorted_table.index),
            [
                ("db0", "rf"),
                ("db0", "mlp"),
                ("db0", "cnn"),
                ("db1", "lr"),
            ],
        )

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
