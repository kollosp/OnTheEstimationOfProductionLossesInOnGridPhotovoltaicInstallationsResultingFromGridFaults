if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
import os, sys
import logging

logger = logging.getLogger(__name__)

WINDOW_LEN_SEC = 60*5 # 5 mins
DAY_LEN = 24*60*60
WINDOWS_PER_DAY = DAY_LEN // WINDOW_LEN_SEC

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

if __name__ == "__main__":
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-1])
    file = os.path.join(file_path, "orgdataset.csv")
    logger.info(f"Loading file: {file}")
    df = pd.read_csv(file, delimiter=",", header=None,names=["timestamp", "id", "Power", "L1V", "L2V", "L3V"])
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.loc[df["L1V"] < 210, "L1V"] = np.nan
    df.loc[df["L2V"] < 210, "L2V"] = np.nan
    df.loc[df["L3V"] < 210, "L3V"] = np.nan
    print("Dataset size before resampling: ", len(df))
    unique_ids = df["id"].unique()
    dfs = []
    for id in unique_ids:
        d = df.loc[df['id'] == id]
        print(f"bResampling id = {id}: ", len(d))
        d.index = d["timestamp"]
        d.drop(columns="id", inplace=True)
        d = d.resample("5min").agg({"Power": "mean", "L1V": "max", "L2V": "max", "L3V": "max"})

        print(f"aResampling id = {id}: ", len(d))
        d["timestamp"] = d.index
        d["id"] = id
        d["Power"].fillna(0, inplace=True)
        # d["L1V"] = np.nan
        # d["L2V"] = np.nan
        # d["L3V"] = np.nan

        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)

    print("Dataset size after resampling: ", len(df))

    df.insert(0, 'Grid OV Fault', df[["L1V", "L2V", "L3V"]].max(axis=1) > 253)
    df.insert(0, 'MountedPower', 0)

    df.loc[df['id'] == 6, 'MountedPower'] = 9.9
    df.loc[df['id'] == 159, 'MountedPower'] = 8.865
    df.loc[df['id'] == 17, 'MountedPower'] = 7.7
    df.loc[df['id'] == 87, 'MountedPower'] = 25

    df.loc[df['id'] == 6, 'Latitude'] = 53.687
    df.loc[df['id'] == 159, 'Latitude'] = 51.244
    df.loc[df['id'] == 17, 'Latitude'] = 51.236
    df.loc[df['id'] == 87, 'Latitude'] = 50.770

    df.loc[df['id'] == 6, 'Longitude'] = 15.172
    df.loc[df['id'] == 159, 'Longitude'] = 16.770
    df.loc[df['id'] == 17, 'Longitude'] = 17.155
    df.loc[df['id'] == 87, 'Longitude'] = 17.355

    # fix timezone changes made due to polish legals
    winter_time = [
        ["2020-10-25", "2021-03-28"],
        ["2021-10-31", "2022-03-22"],
        ["2022-10-30", "2023-03-26"],
        ["2023-10-29", "2024-03-29"]
    ]

    df["timestamp"] = df["timestamp"]- pd.DateOffset(hours=2)
    for wt in winter_time:
        c = df["timestamp"].between(wt[0], wt[1])
        df.loc[c, 'timestamp'] =  df.loc[c, 'timestamp'] + pd.DateOffset(hours=1)

    # filter incorrect values in power column
    df.loc[df["Power"] > df["MountedPower"] *1.2, 'Power'] = 0
    # compute normalized power
    df["NormalizedPower"] = df["Power"] / df["MountedPower"]

    # df['Grid OV Fault'] = df[["L1V", "L2V", "L3V"]].max(axis=1) >= 253
    # pivot table and 5min aggregation
    df["timestamp unix"] = df['timestamp'].astype('datetime64[s]').astype('int')
    df["timestamp unix"] = df["timestamp unix"] - df["timestamp unix"].min()
    df["Bin"] = df["timestamp unix"] // WINDOW_LEN_SEC
    # select all columns. Not selected columns will be removed from dataframe
    # print_full(df[["id", "Bin"]][2*288:].head(288))
    df = df.groupby(["id", "Bin"], observed=True).agg({
        'timestamp': 'min',
        'Power': 'mean',
        'L1V': 'max',
        'L2V': 'max',
        'L3V': 'max',
        'Latitude': 'max',
        'Longitude': 'max',
        'Grid OV Fault': 'max',
        'MountedPower': 'max',
        'NormalizedPower': 'mean',
    })

    # print_full(df[288:].head(288))

    dfs = [df.loc[[id]] for id in df.index.get_level_values(0).unique()]
    for i, idf in enumerate(dfs):
        idf.index  = idf.index.droplevel(0)
        idf.rename(columns={column: f"{i}_{column}" for column in idf.columns}, inplace=True)

    df = dfs[0]
    for i, idf in enumerate(dfs[1:]):
        df = df.join(idf)

    # end of pivot table end 5min aggregation
    # dfs = [df.agg({'timestamp': 'max', 'Power':'mean','L1V':'max','L2V':'max','L3V':'max'}) for df in dfs]
    df["timestamp"] = df[[f"{i}_timestamp" for i in range(len(dfs))]].min(axis=1)

    df.index = df["timestamp"]
    df.drop(columns=[f"{i}_timestamp" for i in range(len(dfs))] + ["timestamp"], inplace=True)

    # for dataset which span over shorter periods that the longest one non values appear. For configuration data remove it
    meta_columns = ["Latitude", "Longitude", "MountedPower"]
    for i, _ in enumerate(unique_ids):
        for mc in meta_columns:
            print(f"{i}_{mc}", max(df[f"{i}_{mc}"]), end=" | ")
            df[f"{i}_{mc}"] = max(df[f"{i}_{mc}"][~np.isnan(df[f"{i}_{mc}"])])
            print(f"{i}_{mc}", max(df[f"{i}_{mc}"]))

    file_out = os.path.join(file_path, "dataset.csv")
    df.to_csv(file_out)
