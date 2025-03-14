if __name__ == "__main__": import __config__
import os
from Analysis import Analysis
from Analysis import TEXTS
from Simulation import Simulation

from os.path import isfile, join, isdir
from matplotlib import pyplot as plt
import logging
import os
import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor
from sktime.forecasting.arima import AutoARIMA

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
                    prog='Generic search validation',
                    description='Reads cm/..._result.csv files located in given directory',
                    epilog='---')

parser.add_argument("-i", "--fileIndex", help="if passed program will use this value to look in cm directory instead of generating cin query",
                    type=str)
parser.add_argument("-d", "--dataset", help="validation dataset (installation) Id",
                    type=int, default=0)

def ask_for_filepath(selection=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = "/".join(dir_path.split("/")[:-1] + ["cm"])
    logger.info(dir_path)
    onlydirs = [f for f in listdir(file_path) if isdir(join(file_path, f)) and not "validation" in f]

    onlydirs.sort()

    if selection is not None:
        file = selection
    else:
        for i, name in enumerate(onlydirs):
            logger.info(f' => {i}. {name}')
        file = input("Enter file number (or enter multiple files using ','): ")

    if "," in file:
        file = [int(f) for f in file.split(",")]
    else:
        file = [int(file)]

    return [file_path + "/" + onlydirs[f] for f in file], [onlydirs[f].split("_")[-1] for f in file]

def get_lstm_model(self, installation_id=None, y_fit=None, model_parameters={}, fit_model=True):
    # model = LSTMFCNRegressor()
    model = AutoARIMA(
        sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True
    )

    if fit_model:  # fit only if it is needed (if fot data is defined)
        model.fit(y=y_fit)

    return model

def lastNPeriods_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "N": [1, 30, True],
        "window_size_days_10": [4,4,True]
    })

def seaippf_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "Bins": [20, 40, True],
        "window_size_days_10": [30, 30, True], # at 31 elbow
        "conv2D_shape_factor": [0, 0.05, False],
        "regressor_degrees": [10, 15, True],
        "bandwidth": [0.01, 0.1, False],
        "iterations": [1, 6, True],
        "iterative_regressor_degrees": [10, 15,True]
    })

def lstm_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [20, 60, True]
    })

def main(args):
    prone_installation_id = 3
    # healthy_installation_id = args.dataset
    limit_voltage = 256

    a = Analysis()

    logging.basicConfig(level=logging.INFO)
    logger.info('Started')
    paths, models = ask_for_filepath(args.fileIndex)
    print(paths, models)
    for path, model in zip(paths, models):
        #remove last "/" if given for concatenation purposes
        if path[-1] == "/":
            path = path[:-1]
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        healthy_installation_id = df["healthy_installation_id"].max()
        df.sort_values(by="value", inplace=True)
        param_columns = [col for col in df.columns if not any(v in col for v in ["value", "_std", "_mean"])]
        logger.info(f"Current model ={model}")
        param_series = df[param_columns].iloc[0]
        logger.info(str(param_series))

        if not "_validation" in path:
            file_prefix = path + "_validation"
        else:
            file_prefix = path
        model_factories = {
            "seaippf": Analysis.get_seaippf_model,
            "nlastperiods": Analysis.get_nlastperiods_model
        }

        if not model in model_factories:
            logger.error(f"incorrect 'model' selection: {model}. Refer to help")
            continue

        s = Simulation(analysis=a, model_factory=model_factories[model], limit_voltage=limit_voltage,
                       require_minimum_samples_count=18, file_prefix=file_prefix, save_each_image=1,
                       validation=True, prone_installation_id=prone_installation_id, healthy_installation_id=healthy_installation_id)
        s.validation(param_series.to_dict())

        logger.info(f"Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
