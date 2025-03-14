import pandas as pd
from typing import List
from sktime.forecasting.model_selection import SlidingWindowSplitter


from utils.Plotter import Plotter

def series_to_Xy(data: pd.Series):
    return (data.index.to_numpy().astype(int)/10**9).astype(int).reshape(-1,1), data.to_numpy()

class Metric:
    def __init__(self, metric, name):
        self._metric = metric
        self._name = name
    def __call__(self, *args, **kwargs):
        return self._metric(*args, **kwargs)
    def __str__(self):
        return self._name

class Experimental:
    def __init__(self):
        self._predictions = None
        self._dataset = None
        self._X, self._y = None, None
        self._models = []
        self._metrics = []

    @property
    def predictions(self):
        return self._dataset

    @property
    def dataset(self):
        return self._dataset

    @property
    def models(self):
        return self._models

    @property
    def metrics(self):
        return self._metrics

    def register_dataset(self, dataset: pd.DataFrame):
        self._dataset = dataset
        self._X, self._y = series_to_Xy(self._dataset)
        print(f"Dataset Registered. Shape: X={self._X.shape}, y={self._y.shape}")

    def register_model(self, model):
        self._models.append(model)
    def register_models(self, models : List):
        for model in models:
            self._models.append(model)

    def register_metric(self, metric, name):
        self._metrics.append(Metric(metric, name))

    def register_metrics(self, metrics: List):
        """
        Register multiple metrics
        :param metrics: a list of tuples containing [(callable, name), (callable, name), ...]
        """
        for metric in metrics:
            self._metrics.append(Metric(metric[0], metric[1]))

    def predict(self, forecast_horizon = 288, batch = 288, window_length = 288 * 30, early_stop=None):
        X, y = self._X, self._y

        fh = [forecast_horizon]
        cv = SlidingWindowSplitter(window_length=window_length, fh=fh, step_length=len(fh))

        # initialize prediction results
        forecast_start_point = forecast_horizon + window_length - 1
        predictions = [[0] * len(y) for _ in self._models]
        splits = cv.get_n_splits(y)

        # make cross validation using SlidingWindowSplitter
        for i, (train, test) in enumerate(cv.split(y)):
            # print(train, test)
            if i % batch == 0:
                print(f"Batch learning. Iter: {i} ~ {int(100 * i/splits)}%")
                # print(f"    Train len: {len(train)}, Test len {len(test)}")
                for model in self._models:
                    # print("y[train]", self._dataset[train])
                    model.fit(y=self._dataset[train])

            for j, model in enumerate(self._models):
                prediction = model.predict(fh=self._dataset[test].index)

                s = forecast_start_point + i
                predictions[j][s:s + len(prediction)] = prediction
                # print(s, prediction, predictions[j][s:s + len(prediction)] )

            if early_stop is not None and i > early_stop:
                print(f"Early stop on i={i} > {early_stop}")
                break

        # create metrics dataframe
        metrics_results = {}
        for metric in self._metrics:
            metrics_results[f"{metric}"] = []
            for model, prediction in zip(self._models, predictions):
                result = metric(prediction[forecast_start_point:], y[forecast_start_point:])
                metrics_results[f"{metric}"].append(result)

        metrics_results["Models"] = [str(m) for m in self._models]
        metrics_results = pd.DataFrame(metrics_results)
        metrics_results.set_index("Models", inplace=True)

        d = {str(model): prediction for model, prediction in zip(self._models, predictions)}
        d["data"] = self._dataset.values
        self._predictions = pd.DataFrame.from_dict(d)
        self._predictions.index = self._dataset.index
        self._predictions = self._predictions[forecast_start_point:]
        return self._predictions, metrics_results, forecast_start_point

    def statistics(self):
        pass

    def plot(self):
        plotter = Plotter(self._X[:, 0], [self._dataset.to_numpy(),
                                                *[self._predictions[column].to_numpy() for column in self._predictions.columns]], debug=False)
        return plotter.show()

