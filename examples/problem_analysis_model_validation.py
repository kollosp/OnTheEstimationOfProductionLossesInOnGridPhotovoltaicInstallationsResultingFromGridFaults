if __name__ == "__main__": import __config__
import os
from Analysis import Analysis
from Analysis import TEXTS
from Simulation import Simulation
from problem_analysis_model_configs import VALIDATION_MODEL_CONFIGS, get_model_from_dirname

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

parser.add_argument("-i", "--fileIndex", help="cm directory selection: index, comma-separated indexes, or 'all' for every eligible result directory",
                    type=str)
parser.add_argument("-d", "--dataset", help="validation dataset (installation) Id",
                    type=int, default=0)

RESULT_STAT_SUFFIXES = ("_std", "_mean", "_max", "_min")


def validation_exists(file_path, name):
    return isdir(join(file_path, f"{name}_validation"))

def ask_for_filepath(selection=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = "/".join(dir_path.split("/")[:-1] + ["cm"])
    logger.info(dir_path)
    onlydirs = [
        f for f in listdir(file_path)
        if isdir(join(file_path, f)) and not "validation" in f and get_model_from_dirname(f) is not None
    ]

    onlydirs.sort()


    if selection is not None:
        file = selection
    else:
        for i, name in enumerate(onlydirs):
            logger.info(f' => {i}. {name} validation_exists={validation_exists(file_path, name)}')
        file = input("Enter file number, multiple files using ',', or 'all': ")

    if file == "all":
        file = [i for i, name in enumerate(onlydirs) if not validation_exists(file_path, name)]
    elif "," in file:
        file = [int(f) for f in file.split(",")]
    else:
        file = [int(file)]

    return [file_path + "/" + onlydirs[f] for f in file], [get_model_from_dirname(onlydirs[f]) for f in file]

def main(args):
    prone_installation_id = 3
    # healthy_installation_id = args.dataset
    limit_voltage = 256

    a = Analysis()

    logging.basicConfig(level=logging.INFO)
    logger.info('Started')
    paths, models = ask_for_filepath(args.fileIndex)
    print("Selected validation inputs:")
    for i, (path, model) in enumerate(zip(paths, models)):
        print(f" => {i}. model={model}, directory={os.path.basename(path)}")
    for path, model in zip(paths, models):
        #remove last "/" if given for concatenation purposes
        if path[-1] == "/":
            path = path[:-1]
        df = pd.read_csv(os.path.join(path, "result.csv"), index_col=0)
        healthy_installation_id = df["healthy_installation_id"].max()
        df.sort_values(by="value", inplace=True)
        param_columns = [
            col for col in df.columns
            if col != "value" and not col.endswith(RESULT_STAT_SUFFIXES)
        ]
        logger.info(f"Current model ={model}")
        param_series = df[param_columns].iloc[0]
        logger.info(str(param_series))

        if not "_validation" in path:
            file_prefix = path + "_validation"
        else:
            file_prefix = path
        if not model in VALIDATION_MODEL_CONFIGS:
            logger.error(f"incorrect 'model' selection: {model}. Refer to help")
            continue

        model_factory, extra_simulation_kwargs = VALIDATION_MODEL_CONFIGS[model]
        simulation_kwargs = {
            "analysis": a,
            "model_factory": model_factory,
            "limit_voltage": limit_voltage,
            "require_minimum_samples_count": 18,
            "file_prefix": file_prefix,
            "validation": True,
            "prone_installation_id": prone_installation_id,
            "healthy_installation_id": healthy_installation_id,
        }
        simulation_kwargs.update(extra_simulation_kwargs)

        s = Simulation(**simulation_kwargs)
        s.validation(param_series.to_dict())

        logger.info(f"Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
