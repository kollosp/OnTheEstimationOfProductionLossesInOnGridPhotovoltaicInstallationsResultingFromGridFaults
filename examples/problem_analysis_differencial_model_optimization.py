if __name__ == "__main__": import __config__
import os
from Analysis import Analysis
from Analysis import TEXTS
from Simulation import Simulation
from problem_analysis_model_configs import MODEL_CHOICES, MODEL_CONFIGS

from matplotlib import pyplot as plt
import logging
import os
import sys
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from itertools import product
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressor
from sktime.forecasting.arima import AutoARIMA

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
                    prog='Generic search',
                    description='Create cm/..._result.csv file for selected model',
                    epilog='---')
parser.add_argument("-m", "--model", help="A model that is selected to evaluation", choices=MODEL_CHOICES,
                    required=True,
                    nargs="+",
                    type=str)
parser.add_argument("-d", "--dataset", help="validation dataset (installation) Id",
                    type=int, nargs="+", default=[0])
parser.add_argument("--dry-run", "--dry_run", help="Log planned runs without starting simulations",
                    action="store_true", dest="dry_run")

def get_simulation_kwargs(model, dataset):

    prone_installation_id = 3
    healthy_installation_id = dataset
    limit_voltage = 256

    model_factory, evaluate_model, extra_simulation_kwargs = MODEL_CONFIGS[model]
    simulation_kwargs = {
        "model_factory": model_factory,
        "limit_voltage": limit_voltage,
        "require_minimum_samples_count": 18,
        "file_prefix": model,
        "date_postfix": f"db{healthy_installation_id}",
        "healthy_installation_id": healthy_installation_id,
        "prone_installation_id": prone_installation_id,
    }
    simulation_kwargs.update(extra_simulation_kwargs)
    return evaluate_model, simulation_kwargs

def run_model(model, dataset, args):
    logger.info(f"Preparing model={model}, dataset={dataset}")

    evaluate_model, simulation_kwargs = get_simulation_kwargs(model, dataset)
    log_simulation_kwargs = {
        key: value for key, value in simulation_kwargs.items() if key != "model_factory"
    }
    logger.info(
        "Simulation configuration for model=%s, dataset=%s: model_factory=%s, kwargs=%s",
        model,
        dataset,
        simulation_kwargs["model_factory"].__name__,
        log_simulation_kwargs,
    )

    if args.dry_run:
        logger.info(f"Dry run: skipping simulation for model={model}, dataset={dataset}")
        return

    a = Analysis()

    simulation_kwargs["analysis"] = a
    s = Simulation(**simulation_kwargs)
    logger.info(f"Starting simulation for model={model}, dataset={dataset}")
    evaluate_model(s, args)
    logger.info(f"Finished simulation for model={model}, dataset={dataset}")

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(levelname)s: %(message)s")
    runs = list(product(args.model, args.dataset))
    logger.info(
        "Requested %s run(s): models=%s, datasets=%s, dry_run=%s",
        len(runs),
        args.model,
        args.dataset,
        args.dry_run,
    )

    if args.dry_run:
        logger.info("Dry run enabled; simulations will not be started.")
        for model, dataset in runs:
            run_model(model, dataset, args)
        logger.info("Dry run done.")
        return

    if len(runs) == 1:
        model, dataset = runs[0]
        logger.info("Running a single simulation without Pool.")
        run_model(model, dataset, args)
        logger.info("Done.")
        return

    logger.info(f"Running {len(runs)} simulations in parallel using Pool.")
    with Pool() as pool:
        pool.starmap(run_model, [(model, dataset, args) for model, dataset in runs])
    logger.info("Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
