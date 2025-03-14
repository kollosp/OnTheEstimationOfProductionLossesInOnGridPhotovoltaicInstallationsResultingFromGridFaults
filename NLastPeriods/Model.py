from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sklearn.metrics import r2_score
from utils import Solar
from utils.Plotter import Plotter
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List

import warnings


class Model(BaseForecaster):
    """This class implements model with different prediction method. Not iterative. """
    _tags = {
        "requires-fh-in-fit": False
    }

    def __init__(self, N=3, period=pd.DateOffset(day=1)):
        super().__init__()
        self.N = int(N)
        self.period = period

    def _fit(self, y, X=None, fh=None):
        self.y_ = y
        self.x_time_delta_ = (y.index[-1] - y.index[0]) / len(y)
        self.y_max_ = max(y.values)

    def _predict(self, fh, X):
        if not any(type(t) == pd._libs.tslibs.timestamps.Timestamp for t in fh):
            ts = np.array([i*self.x_time_delta_ + self.cutoff for i in fh]).flatten()
        else:
            ts = fh.to_numpy()
        return self.in_sample_predict(X,ts)

    def in_sample_predict(self, X, ts):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            prediction = np.zeros(len(ts))
            for i, timestamps in enumerate(ts):
                lower_b = timestamps - self.x_time_delta_
                upper_b = timestamps + self.x_time_delta_
                _y = np.array([self.y_[lower_b - pd.DateOffset(day=n):upper_b - pd.DateOffset(day=n)].mean() for n in range(1, self.N+1)])
                # print(_y, _y.mean())
                prediction[i] = _y.mean()

            prediction[np.isnan(prediction)] = 0
            return pd.Series(data=prediction, index=ts)

    def __str__(self):
        return "NLastDays (" + str(self.get_params()) + ")"

    def plot(self):
        pass