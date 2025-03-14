if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
import os
from datetime import date
from matplotlib import pyplot as plt
import matplotlib.dates as md
from math import ceil, sqrt
from datasets import utils
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from SEAIPPF.Transformers.TransformerSimpleFiltering import TransformerSimpleFiltering

from utils.Evaluate import Evaluate
from utils.Plotter import Plotter
from utils import Solar
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sktimeSEAPF.Optimized import Optimized

def print_full(x):
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 2000)
    # pd.set_option('display.float_format', '{:20,.2f}'.format)
    # pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

if __name__ == "__main__":
    installation = 1
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    # self.full_data = self.full_data[30:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df.drop(columns=["timestamp"], inplace=True)
    # print_full(df[["0_Power"]].head(288))
    latitude_degrees = df[f"{installation}_Latitude"][0]
    longitude_degrees = df[f"{installation}_Longitude"][0]
    # print_full(df.index.astype('datetime64[s]'))
    timestamps = df.index.astype('datetime64[s]').astype('int')

    elevation = Solar.elevation(Optimized.from_timestamps(timestamps), latitude_degrees,
                                longitude_degrees) * 180 / np.pi
    #d = pd.DataFrame({
    #    "power": df[f"{installation}_Power"],
    #    "elevation": elevation
    #}, index = df.index)
    #print_full(df[[f"{installation}_Power"]][288:].head(288))
    # print("First 288 rows")


    plotter = Plotter(df.index, [df[i] for i in [f"{j}_Power" for j in range(0,4)]] + [elevation], list_of_line_names=[f"{j}_Power" for j in range(0,4)], debug=False)
    plotter.show()
    plt.show()


