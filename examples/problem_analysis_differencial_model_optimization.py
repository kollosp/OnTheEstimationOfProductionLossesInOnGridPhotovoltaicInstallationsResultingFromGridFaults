if __name__ == "__main__": import __config__
import os
from Analysis import Analysis
from Analysis import TEXTS
from Simulation import Simulation

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
                    prog='Generic search',
                    description='Create cm/..._result.csv file for selected model',
                    epilog='---')
parser.add_argument("-m", "--model", help="A model that is selected to evaluation", choices=["seaippf", "nlastperiods", 'lstm'],
                    required=True,
                    type=str)
parser.add_argument("-d", "--dataset", help="validation dataset (installation) Id",
                    type=int, default=0)

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
        "window_size_days_10": [20, 40, True], # at 31 elbow
        "conv2D_shape_factor": [0, 0.05, False],
        "regressor_degrees": [8, 15, True],
        "bandwidth": [0.01, 0.1, False],
        "iterations": [1, 6, True],
        "iterative_regressor_degrees": [10, 15,True],
        "stretch": [1, 1, True]
    })

def lstm_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [20, 60, True]
    })

def main(args):
    prone_installation_id = 3
    healthy_installation_id = args.dataset
    limit_voltage = 256

    a = Analysis()
    if args.model == "seaippf":
        logger.info(f"Operating on {args.model}")
        model_factory = Analysis.get_seaippf_model
        s = Simulation(analysis=a, model_factory=model_factory, limit_voltage=limit_voltage,
                       require_minimum_samples_count=18, file_prefix="seaippf", save_each_image=1,
                       date_postfix=f"db{healthy_installation_id}",
                       healthy_installation_id=healthy_installation_id, prone_installation_id=prone_installation_id)
        seaippf_model(s, args)
    elif args.model == "nlastperiods":
        logger.info(f"Operating on {args.model}")
        model_factory = Analysis.get_nlastperiods_model
        s = Simulation(analysis=a, model_factory=model_factory, limit_voltage=limit_voltage,
                       require_minimum_samples_count=18, file_prefix="nlastperiods",
                       date_postfix=f"db{healthy_installation_id}",
                       healthy_installation_id=healthy_installation_id, prone_installation_id=prone_installation_id)
        lastNPeriods_model(s, args)
    elif args.model == "lstm":
        logger.info(f"Operating on {args.model}")
        s = Simulation(analysis=a, model_factory=get_lstm_model, limit_voltage=limit_voltage,
                       require_minimum_samples_count=18, file_prefix="lstm",
                       date_postfix=f"db{healthy_installation_id}",
                       healthy_installation_id=healthy_installation_id, prone_installation_id=prone_installation_id)
        lstm_model(s, args)

    else:
        logger.error(f"incorrect 'model' selection: {model}. Refer to help")
    logger.info(f"Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
