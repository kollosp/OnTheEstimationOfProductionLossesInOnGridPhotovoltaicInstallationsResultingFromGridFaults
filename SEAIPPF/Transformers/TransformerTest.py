from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import SEAIPPF.image as image
from SEAIPPF.MyPipeline import MyPipline
import numpy as np
from SEAIPPF.RegressorTransformer import RegressorTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures
from math import ceil
import seaborn as sns
from matplotlib import pyplot as plt

"""
Model using this transformer achieves R2 = 0.59 
"""
class TransformerTest(BaseEstimator,TransformerMixin):
    def __init__(self,
                 regressor_degrees=11,
                 iterative_regressor_degrees=11,
                 conv2D_shape_factor = 0.05,
                 iterations=3,
                 stretch=True,
                 sklearn_regressor = LinearRegression()):
        self.iterations = int(iterations)
        self.sklearn_regressor = sklearn_regressor
        self.iterative_regressor_degrees = iterative_regressor_degrees
        self.regressor_degrees = regressor_degrees
        self.stretch = stretch
        self.conv2D_shape_factor = conv2D_shape_factor


    def fit(self, X, y=None):
        return self

    def get_params_info(self):
        params = self.get_params()
        # Default value, min bound, max bound, integrity
        params["conv2Dx_shape_factor"] = (0.1, 0, 1, False)
        params["conv2Dy_shape_factor"] = (0.5, 0, 1, False)
        params["hit_points_neighbourhood"] = (3, 1, 12, True)
        params["regressor_degrees"] = (11, 4, 15, True)
        return params

    def append_pipline_steps(self, pipeline):
        for step in pipeline.steps[1:]:
            self.steps_.append(step)

    def append_step(self, step):
        self.steps_.append(step)

    def statistics(self):
        return "Layers:" + "\n".join([step[0] for step in self.steps_])

    def transform(self, X, y=None):
        self.steps_ = []
        self.shape_ = X.shape
        shape = X.shape[0]

        initial = MyPipline([
            # ('conv2Dy', image.Convolution(kernel=np.ones((1, int(self.conv2Dy_shape_factor * shape) + 1)))),
            # ('iConv2DX', image.Convolution(kernel=np.ones((int(self.conv2Dx_shape_factor * shape) + 1, 1)))),
            ('iConv2D', image.Convolution(kernel=np.ones(
                (int(self.conv2D_shape_factor * shape) + 1, int(self.conv2D_shape_factor * shape) + 1)))),
        ])
        X = initial.fit_transform(X)
        self.append_pipline_steps(initial)
        for i in range(self.iterations):
            iterative = MyPipline([
                (f'x{i+1}-LocalMax', image.LocalMaximumsAlongAxis()),
                (f'x{i+1}-TakeLast', image.TakeLastAlongAxis()),
                (f'x{i+1}-PolyReg(degree={self.iterative_regressor_degrees})', RegressorTransformer(regressor=make_pipeline(
                    PolynomialFeatures(int(self.iterative_regressor_degrees)),
                    self.sklearn_regressor
                ))),
                (f'x{i+1}-Dilate', image.Dilation(kernel_size=(5,5))),
                (f'x{i+1}-Conv2D', image.Convolution(kernel=np.ones(
                    (int(self.conv2D_shape_factor * shape) + 1, int(self.conv2D_shape_factor * shape) + 1)))),
            ])
            mask = iterative.fit_transform(X)
            self.append_pipline_steps(iterative)
            X = X*mask
            self.append_step((f'x{i+1}-Mask', X))

        finalizer = MyPipline([
            (f'eLocalMax', image.LocalMaximumsAlongAxis(only_max=True)),
            (f'eReg(deg={self.regressor_degrees})', RegressorTransformer(regressor=make_pipeline(
                PolynomialFeatures(int(self.regressor_degrees)),
                self.sklearn_regressor
            ))),
            ("eStretch", image.RemoveEmptyAlongAxis(stretch=self.stretch))
        ])
        X = finalizer.fit_transform(X)
        self.append_pipline_steps(finalizer)
        return X

    def __str__(self):
        return f"TransformerTest({self.get_params()})"

    def plot(self, rows = 3, ax=None, fig=None):
        count = sum(name[0] != "_" for name, _ in self.steps_)
        axis = ceil(count / rows ), rows

        if ax is None or fig is None:
            fig, ax = plt.subplots(axis[1], axis[0])

        fig.suptitle("Model's transformation chain steps applied to the heatmap")
        _ax = [ax[i, j] for j in range(axis[0]) for i in range(axis[1])]

        i = 0
        for (name, result) in self.steps_:
            if name[0]!="_":
                _ax[i].set_title(f"{i+1}: {name}", fontdict={"fontsize": 8})
                sns.heatmap(result.reshape(self.shape_), cmap='viridis', ax=_ax[i])
                _ax[i].invert_yaxis()
                _ax[i].set_xticks([])
                _ax[i].set_yticks([])
                i += 1
        while i<len(_ax):

            _ax[i].remove()
            i += 1
        return fig, ax