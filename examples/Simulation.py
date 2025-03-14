from scipy.optimize import differential_evolution
from utils.Evaluate import Evaluate
from utils.Experimental import Experimental
from utils.ExecutionTimer import ExecutionTimer
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.Plotter import Plotter
from utils import Solar
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, max_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sktimeSEAPF.Optimized import Optimized
from datetime import datetime
import numpy as np
import pandas as pd

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

class Simulation():
    def __init__(self, analysis, model_factory, prone_installation_id=3, healthy_installation_id=0,
                 limit_voltage=256, require_minimum_samples_count=18, file_prefix = "", date_postfix="", save_each_image=None, validation=False):
        self.analysis = analysis
        self.model_factory = model_factory
        self.prone_installation_id = prone_installation_id
        self.healthy_installation_id = healthy_installation_id
        self.limit_voltage = limit_voltage
        self.require_minimum_samples_count = require_minimum_samples_count
        self.current_best_value= None
        self.current_best_model_parameters = {}
        self.save_each_image = save_each_image
        self.result_list = []
        today = datetime.now()
        if validation: #use same timedate directory name as was used during evaluation
            # self.txt_datetime = file_prefix + "_results.csv"
            self.output_dir =  file_prefix
        else: # set timedate directory name
            # self.txt_datetime = "cm/" + today.strftime("%Y%m%d-%H%M%S") + "_" + file_prefix + "_results.csv"
            self.output_dir = "cm/" + today.strftime("%Y%m%d-%H%M%S") + (f"_{date_postfix}" if date_postfix != "" else "")+ "_" + file_prefix


        self.current_iteration = 0
        self.model = None

    def build_result_list(self, value, model_parameters, results_df):
        results_df_means = results_df.mean()
        results_df_std = results_df.std()
        results_df_max = results_df.max()
        results_df_min = results_df.min()
        append = {
            "value": value,
        }
        for k in model_parameters.keys():
            append[k] = model_parameters[k]
        for k in results_df_means.index:
            append[k + "_mean"] = results_df_means[k]
            append[k + "_std"] = results_df_std[k]
            append[k + "_max"] = results_df_max[k]
            append[k + "_min"] = results_df_min[k]
        self.result_list.append(append)
        return self.result_list

    def save_best_model(self, value, model_parameters, results_df):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.build_result_list(value, model_parameters, results_df)

        new_best_found = False
        if self.current_best_value is None or value < self.current_best_value:
            self.current_best_value = value
            self.current_best_model_parameters = model_parameters
            print("New best found at config: ", model_parameters, "with MAE: ", value)
            new_best_found = True
        else:
            print(f"Current value {value} is worse than current best {self.current_best_value}")

        if self.current_iteration % 10 == 0 or new_best_found:
            file =  self.output_dir + f"/result.csv"
            df = pd.DataFrame(self.result_list)
            df["healthy_installation_id"] = self.healthy_installation_id
            df["prone_installation_id"] = self.prone_installation_id
            df.to_csv(file)
            print(f"Saved in file: {file}")

        if new_best_found: #or (self.save_each_image is not None and self.current_iteration % self.save_each_image == 0):
            self.model.plot()
            file = self.output_dir + f"/images_{self.current_iteration}.pdf"
            pp = PdfPages(file)
            for i in plt.get_fignums():
                f = plt.figure(i)
                # plt.savefig(self.img_files + 'figure%d.png' % i)
                pp.savefig(f)
            pp.close()

            for i in plt.get_fignums():
                f = plt.figure(i)
                f.clear()
                plt.close(f)
            print(f"Images saved in file: {file}")

        self.current_iteration += 1
        return value

    def __call__(self, params=[], param_names=None):
        model_parameters = {key: value for key, value in zip(param_names, params)}
        print(f"Next interation: {self.current_iteration}", model_parameters)
        df = self.call(model_parameters)

        return self.save_best_model(df["mae"].mean(), model_parameters, df)

    def call(self, model_parameters, validation=False):
        window_size = int(model_parameters.get("window_size_days_10", 12) * 288 * 10)
        y_tests, y_trains, y_trues = self.analysis.get_experiment_dataset(self.prone_installation_id,
                                                                          self.healthy_installation_id,
                                                                          self.limit_voltage, window_size,
                                                                          require_minimum_samples_count=self.require_minimum_samples_count,
                                                                          validation=validation)
        results = {
            "i": [],
            "mse": [],
            # "r2": [],
            "mae": [],
            "time": [],
            "mape": [],
            "rmse": [],
            "max_error": [],
            "count": [],
            "NEE Forecast": [],  # NEE = Norm. Energy Eqvi.
            "NEE Lost": [],
            "NEE AE": [],  # absolute error sum(abs(e1-e2))
            "NEE All Day": [],
            "NEE Lost factor": [],  # AE / All Day
        }
        for i, (y_test, y_train, y_true) in enumerate(zip(y_tests, y_trains, y_trues)):
            timer = ExecutionTimer()

            with timer as t:
                y_true = y_true[~np.isnan(y_true)]
                self.model = self.model_factory(
                    self.analysis,
                    installation_id=self.healthy_installation_id,
                    model_parameters=model_parameters,
                    y_fit=y_train
                )
                y_pred = self.model.predict(fh=y_true.index)

            results["i"].append(i)
            results["time"].append(timer.seconds_elapsed)
            results["mae"].append(mean_absolute_error(y_pred, y_true))
            results["mse"].append(mean_squared_error(y_pred, y_true))
            results["mape"].append(mean_absolute_percentage_error(y_pred, y_true))
            results["rmse"].append(root_mean_squared_error(y_pred, y_true))
            results["max_error"].append(max_error(y_pred, y_true))
            # results["r2"].append(r2_score(y_pred, y_true))
            results["count"].append(len(y_pred))
            el = y_true.sum() / 12
            results["NEE Lost"].append(el)  # in [kWh / kW]
            ef = y_pred.sum() / 12
            results["NEE Forecast"].append(ef)  # in [kWh / kW]
            results["NEE AE"].append(sum(abs(y_true - y_pred)) / 12)  # in [kWh / kW]
            ad = (y_true.sum() + y_test.sum()) / 12
            results["NEE All Day"].append(ad)  # in [kWh / kW]
            results["NEE Lost factor"].append(el / ad)  # in [kWh / kW]

            txt = [f"Progress: {100 * i / len(y_tests):.0f}% ", f"Len: {len(y_tests)}"]
            for key in results.keys():
                txt.append(f"{key}: {results[key][-1]:.2f}")
            print("\r" + ", ".join(txt), end="")

        print("")
        df = pd.DataFrame(results, index=results["i"])

        return df

    def validation(self, model_parameters):
        print(model_parameters)
        df = self.call(model_parameters, validation=True)
        return self.save_best_model(df["mae"].mean(), model_parameters, df)

    def generic_evaluate(self, parameters={}):
        """
        parameters: a dictionary defines args of objective function ex.
        {"param1": [<lower bound>, <upper bound>, <integrity>], "param2":...}

        """
        bounds = []
        integrality = []
        for key in parameters.keys():
            bounds.append(parameters[key][0:2])
            integrality.append(parameters[key][2])

        results = differential_evolution(self, bounds=bounds, integrality=integrality, args=[parameters.keys()])
        return results