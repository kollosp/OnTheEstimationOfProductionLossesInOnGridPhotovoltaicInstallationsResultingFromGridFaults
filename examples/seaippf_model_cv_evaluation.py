if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from SEAIPPF.Transformers.TransformerSimpleFiltering import TransformerSimpleFiltering
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sktime.forecasting.model_evaluation import evaluate
from utils.Plotter import Plotter
from sklearn.metrics import r2_score
from sktime.split import SlidingWindowSplitter
from sklearn.neural_network import MLPRegressor
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from Analysis import load_data
from Analysis import Analysis

import random

def combine_evaluation_data(evaluation_return_data):
    data_columns = ["y_test","y_pred"]
    for dc in data_columns:
        if not dc in evaluation_return_data.columns:
            raise ValueError("Incorrect evaluation_return_data passed. Ensure return_data=True in sktime.forecasting.model_evaluation.evaluate function")

    #clear statistic table and data
    train_test_data = evaluation_return_data[data_columns]
    evaluation_return_data = evaluation_return_data.drop(columns=data_columns + ["y_train"])

    data_columns_series = []
    for dc in data_columns:
        y = pd.concat(train_test_data[dc].to_list(), axis=1)

        # y.fillna(0, inplace=True)
        y.columns = [f"{dc}{i}" for i, _ in enumerate(y.columns)]
        y = y.agg(np.nanmax, axis="columns")
        data_columns_series.append(y)
    df = pd.DataFrame({k:i.values for k,i in zip(data_columns, data_columns_series)})
    return df, evaluation_return_data

class Progress:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.current_iteration = 0
    def __call__(self, *args, **kwargs):
        self.current_iteration += 1
        print(f"\rProgress: {100*self.current_iteration / self.max_iterations:.0f}%", end="")

def evaluation(model):
    a = Analysis()
    data,_ = a.get_experiment_dataset(
        prone_installation_id=3,
        healthy_installation_id=0,
        limit_voltage=256,
        require_minimum_samples_count=18,
        split_data=False)

    start=288*110 * random.randint(1, 4)
    window_length = 288 * 365
    end=start + window_length + 288*31
    data = data[start:end]
    data[data > 15] = 0

    cv = SlidingWindowSplitter(window_length=window_length, step_length=288*30, fh=list(range(1,30*288+1)))

    print(f"Evaluation started. number of tests: {cv.get_n_splits(data)}")
    p = Progress(cv.get_n_splits(data))
    results = evaluate(forecaster=model, y=data, cv=cv, return_data=True, scoring=MeanAbsoluteError(), error_score='raise')
    print("Evaluation ended")
    return_data,results = combine_evaluation_data(results)

    print(results.head())
    print(results.agg(np.mean, axis=0))
    print(results)
    model.fit(data[0:window_length])
    print("R2: ", r2_score(return_data["y_pred"], return_data["y_test"]))
    model.plot()

    plotter = Plotter(return_data.index, [return_data[col].values for col in return_data.columns], debug=False)
    plotter.show()
    plt.show()

if __name__ == "__main__":
    # original approach
    # evaluation(SEAIPPFModel(
    #     latitude_degrees=utils.LATITUDE_DEGREES,
    #     longitude_degrees=utils.LONGITUDE_DEGREES,
    #     x_bins=70,
    #     y_bins=70,
    #     interpolation=True,
    #     enable_debug_params= True,
    #     transformer=TransformerTest(
    #         regressor_degrees=11,
    #         sklearn_regressor = LinearRegression())
    #         #sklearn_regressor = MLPRegressor(hidden_layer_sizes=(15,15), activation="relu",  max_iter=10000, tol=1e-4, verbose=False))
    #         # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(20,20,15,15), activation="relu",  max_iter=4000, verbose=True))
    #         # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(25,20,20), activation="logistic",  max_iter=10000, verbose=True))
    #         # sklearn_regressor = MLPRegressor(hidden_layer_sizes=(5,5,5), activation="logistic",  max_iter=10000, verbose=True)
    # ))

    #from paper images
    evaluation(SEAIPPFModel(
                # latitude_degrees=self.full_data[f"{installation_id}_Latitude"][0],
                # longitude_degrees=self.full_data[f"{installation_id}_Longitude"][0],
                x_bins=40,
                y_bins=40,
                interpolation=True,
                bandwidth=0.09,
                enable_debug_params= True,
                transformer=TransformerTest(
                    regressor_degrees=12,
                    # hit_points_max_iter=5,
                    # hit_points_neighbourhood=0.1,
                    # conv2Dx_shape_factor = 0.05,
                    # kde_r=0.15,
                    # kde_epsilon = 1/4,
                )
            ))












