from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures
from functools import reduce
from scipy.ndimage import gaussian_filter,grey_dilation,grey_erosion,morphological_gradient
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import RBFInterpolator
import copy

def onceKernel(shape, value = 1):
    return np.array([[value] * shape[1]] * shape[0])

class AdaptiveThreshold(BaseEstimator,TransformerMixin):
    def __init__(self, threshold=.8):
        self.threshold = threshold
    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        analyse = Analyse()
        analyse.fit(X)
        threshold = analyse.get_threshold(self.threshold)
        X[X < threshold] = 0
        # X[X >= threshold] = 1

        return X



class Convolution(BaseEstimator,TransformerMixin):
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y=None):
        self.shape_ = X.shape
        return self

    def transform(self, X, y=None):

        if not X.shape == self.shape_:
            X = X.reshape(self.shape_)
        X = signal.convolve2d(X, self.kernel, boundary='symm', mode='same')
        return X.reshape(self.shape_)

class Magnitude(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=2):
        self.gamma = gamma

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return expit(4 * self.gamma * (X - X.mean()))



class RegressorTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shape = X.shape
        c = np.concatenate([[[j], [i], [w]]
                            for i, x in enumerate(X)
                            for j, w in enumerate(x)], axis=1)

        X = np.concatenate([
           # np.array([c[0, i], c[1, i]] * (c[2, i] * 10).astype(int)).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
           np.array([c[0, i], c[1, i]] ).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        ])

        # cls = SVR(C=self.C, epsilon=self.epsilon, degree=self.degree)
        self.regressor.fit(X=X[:, 0:1], y=X[:, 1])

        pred = self.regressor.predict(X=np.array(list(range(0, shape[1]))).reshape(-1, 1))
        pred_arr = np.zeros(shape)
        pred_i = pred.astype(int)
        pred_i[pred_i >= pred_arr.shape[1]] = pred_arr.shape[1] -1
        # print(pred)
        for i in range(0, pred_arr.shape[1]):
            # print(pred_arr.shape[1], i, pred_i[i])
            pred_arr[pred_i[i], i] = 1

        return pred_arr

class Analyse(BaseEstimator,TransformerMixin):
    def __init__(self, bins=100):
        self.bins = bins

    def get_threshold(self, percentage_left):
        """
        percentage_left: percentage values that should left in the histogram after threshold application
        """
        initial_sum = np.sum(self.value_histogram_)
        sum_left_after_removing = initial_sum
        goal_sum = initial_sum * percentage_left

        for i in range(self.bins -1,0,-1):
            if goal_sum < sum_left_after_removing :
                sum_left_after_removing = sum_left_after_removing - self.value_histogram_[i] #remove last bin
            else:
                return self.value_histogram_bins_[i]

        return self.value_histogram_bins_[0]

    def fit(self, X, y=None):
        self.shape_ = X.shape
        self.X_ = X.copy()
        self.value_histogram_, self.value_histogram_bins_ = np.histogram(X.flatten(), bins=self.bins)
        self.along_x_ = np.sum(X, axis=1)
        self.along_y_ = np.sum(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X

    def plot(self):
        fig, ax = plt.subplots(2,2)
        _ax = ax[0,0]

        sns.heatmap(self.X_ , cmap='viridis', ax=_ax)
        _ax.invert_yaxis()
        _ax = ax[1, 1]
        _ax.hist(self.value_histogram_bins_[:-1], self.value_histogram_bins_, weights=self.value_histogram_)
        _ax = ax[0, 1]
        s = list(range(self.X_.shape[0])) #+ [self.X_.shape[0]]
        _ax.barh(s, self.along_x_)

        _ax = ax[1, 0]
        s = list(range(self.X_.shape[1])) #+ [self.X_.shape[1]]
        _ax.bar(s, self.along_y_)
        return fig, ax

class LocalMaximumsAlongAxis(BaseEstimator,TransformerMixin):
    def __init__(self, limit=0.1, only_max=False):
        self.limit = limit
        self.only_max = only_max

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("Threshold.transform: X should have at least 2 dimensions ")
        Z = np.zeros(X.shape)
        # np.set_printoptions(precision=2)
        # np.set_printoptions(suppress=True)
        for i in range(X.shape[1]):
            x = X[:, i]
            derivative2 = np.diff(np.diff(x))  # second derivative. Negative -> Max, Positive Min
            # derivative2 = derivative2 * -1
            extremes = np.diff(np.sign(np.diff(x)))# second derivative. -2 -> Max, +2 -> Min other
            extremes[abs(extremes) < 2] = 0 # clear not sure places
            extremes = extremes * -1 # reverse meaning +2 -> Max, -2 -> Min
            extremes = np.sign(extremes) # move to <-1,1> range
            if self.only_max:
                extremes[extremes < 0] = 0
            # print(derivative2)
            absd = abs(derivative2)
            extremes[absd < self.limit * max(absd) ] = 0
            # print(extremes)
            Z[1:-1, i] = extremes
        return Z

class HitPoints(BaseEstimator,TransformerMixin):
    def __init__(self, max_iter=3, neighbourhood=5, preserve_original_values=False):
        self.max_iter = max_iter
        self.neighbourhood = neighbourhood
        self.preserve_original_values = preserve_original_values

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("HitPoints.transform: X should have at least 2 dimensions ")
        XX = copy.deepcopy(X)

        Z = np.zeros(X.shape)
        #print("Hitpoints.transform,", self.neighbourhood)
        for _ in range(self.max_iter):
            imx = np.argmax(XX, axis=0)
            #print("Hitpoints.transform,", self.neighbourhood, len(imx))
            # print(list(imx))
            for j,index in enumerate(imx):
                if XX[index, j] > 0:
                    Z[index, j] = 1
                    mi = int(index-self.neighbourhood)
                    mx= int(index+self.neighbourhood)
                    mi = 0 if mi < 0 else mi
                    mx = XX.shape[0]-1 if mx > XX.shape[0] else mx

                    XX[mi:mx, j] = 0 # clear neighbourhood

        if self.preserve_original_values:
            return X*Z
        else:
            return Z


class RBFInter(BaseEstimator,TransformerMixin):
    def __init__(self, epsilon=10):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        coordinates = np.argwhere(X == 1)
        xgrid = np.mgrid[0:X.shape[0]:1, 0:X.shape[1]:1]
        xflat = xgrid.reshape(2, -1).T

        rbf = RBFInterpolator(coordinates, np.ones(len(coordinates)), epsilon=self.epsilon, kernel="linear")(xflat)
        rbf = rbf.reshape(X.shape)
        return rbf

class TakeLastAlongAxis(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Z = np.zeros(X.shape)
        for col in range(X.shape[1]):
            y = np.flatnonzero(X[:, col])
            if len(y >= 1):
                Z[y[-1], col] = 1
        return Z

class TakeFirstAlongAxis(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Z = np.zeros(X.shape)
        for col in range(X.shape[1]):
            y = np.flatnonzero(X[:, col])
            if len(y >= 1):
                Z[y[0], col] = 1
        return Z

class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X



class KernelProcess(BaseEstimator, TransformerMixin):
    def __init__(self, kernel="gaussian", epsilon=1/4, max_roots=100, r=None):
        """
        :param kernel: function to estimate density
        :param epsilon:
        :param max_roots: number of points used to estimate density
        """
        self.epsilon = epsilon
        self.kernel = kernel
        self.r = r
        self.max_roots = max_roots

    def fit(self, X, y=None):
        return self

    def relu(self, r):
        y = 1 - self.epsilon * r
        if y >= 0:
            return y
        else:
            return 0

    def multiquadric(self, r):
        return - np.sqrt(1 + self.epsilon * r**2)

    def gaussian(self, r):
        return np.exp(-(self.epsilon * r)**2)

    def transform(self, X, y=None):
        coordinates = np.argwhere(X > 0)
        xgrid = np.mgrid[0:X.shape[0]:1, 0:X.shape[1]:1]
        xflat = xgrid.reshape(2, -1).T

        if self.kernel == "linear":
            f = self.relu
        elif self.kernel == "multiquadric":
            f = self.multiquadric
        elif self.kernel == "gaussian":
            f = self.gaussian
        else:
            f = self.relu

        if self.r is not None:
            Z = np.array([np.sum([X[tuple(c)] * f(np.sqrt(np.sum((c - x) ** 2))) for c in coordinates if
                                  np.sqrt(np.sum((c - x) ** 2)) < self.r]) for x in xflat])
        else:
            Z = np.array([np.sum([X[tuple(c)] * f(np.sqrt(np.sum((c - x) ** 2))) for c in coordinates]) for x in xflat])
        #sqrt((x1-x1)^2 + ... + (xn-xn)^2) - euclidean distance
        Z = Z.reshape(X.shape)
        return Z

class ThresholdAlongAxis(BaseEstimator,TransformerMixin):
    BINARY = 1
    TO_ZERO = 2
    def __init__(self, factor=1, type=BINARY, axis=0):
        self.factor = factor
        self.axis = axis
        self.type = type

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("Threshold.transform: X should have at least 2 dimensions ")

        means = np.mean(X, axis=self.axis)

        if self.axis == 0:
            if self.type == ThresholdAlongAxis.TO_ZERO:
                for i,mean in enumerate(means):
                    X[:, i][X[:, i] < mean * self.factor] = 0
            else:
                for i,mean in enumerate(means):
                    X[:, i][X[:, i] <= mean * self.factor] = 0
                    X[:, i][X[:, i] > mean * self.factor] = 1
        else:
            if self.type == ThresholdAlongAxis.TO_ZERO:
                for i,mean in enumerate(means):
                    X[i, :][X[i, :] < mean * self.factor] = 0
            else:
                for i,mean in enumerate(means):
                    X[i, :][X[i, :] <= mean * self.factor] = 0
                    X[i, :][X[i, :] > mean * self.factor] = 1
        return X

class Erosion(BaseEstimator,TransformerMixin):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("Threshold.transform: X should have at least 2 dimensions ")
        return grey_erosion(X, size=self.kernel_size)

class Sunset(BaseEstimator,TransformerMixin):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("Threshold.transform: X should have at least 2 dimensions ")
        X[:, -self.kernel_size:] = 0
        X[:self.kernel_size, -self.kernel_size:] = 1
        return X

class Sunrise(BaseEstimator,TransformerMixin):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("Threshold.transform: X should have at least 2 dimensions ")
        X[:, :self.kernel_size] = 0
        X[:self.kernel_size, :self.kernel_size] = 1
        return X

class Dilation(BaseEstimator,TransformerMixin):
    def __init__(self, kernel_size=(3,3)):
        self.kernel_size = kernel_size

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        return grey_dilation(X, size=self.kernel_size)

class RemoveEmptyAlongAxis():
    """
    Transformer moves binary mask to remove empty rows at the bottom of X
    """
    def __init__(self, stretch):
        self.stretch = stretch

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        nonzeros = np.sum(X.astype(int), axis=1).nonzero()[0]
        first_non_zero_row = nonzeros[0] #returns tuple of arrays (coordinates in array)
        last_non_zero_row = nonzeros[-1] if self.stretch else X.shape[1] #returns tuple of arrays (coordinates in array)
        shape = X.shape
        # is similar then array will be empty or one dimensional which will lead to error, so simply skip it
        if first_non_zero_row == last_non_zero_row:
            return X

        X = X[first_non_zero_row:last_non_zero_row,:] # removes empty rows -> make array smaller
        ones_occurrence_percent = np.where(np.count_nonzero(X, axis=0)==0, -1, (X.shape[0]-1) - np.argmin(X[::-1,:]==0, axis=0))
        ones_occurrence_percent = ones_occurrence_percent / X.shape[0]
        # set borders
        ones_occurrence_percent[0] = 0
        ones_occurrence_percent[-1] = 0
        #revert original shape
        Z = np.zeros(shape)
        for col in range(Z.shape[1]):
            Z[int(ones_occurrence_percent[col] * Z.shape[0]),col] = 1

        return Z

def opening(mat, size=(3,3)):
    mat = grey_erosion(mat, size=size)
    return grey_dilation(mat, size=size)

def closing(mat, size=(3,3)):
    mat =  grey_dilation(mat, size=size)
    return grey_erosion(mat, size=size)

def np_gradient(x):
    x = np.gradient(x)
    x = np.sqrt(x[0] ** 2 + x[1] ** 2)
    # print(x)
    return x


def make_h_line(shape):
    z = np.zeros(shape)
    z[shape[0]//2,:] = 1
    return z
def make_v_line(shape):
    z = np.zeros(shape)
    z[:, shape[1]//2] = 1
    return z
