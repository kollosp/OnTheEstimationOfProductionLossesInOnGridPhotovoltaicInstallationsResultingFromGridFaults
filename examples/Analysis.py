import pandas as pd
import numpy as np
import os
from datetime import date
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as md
from datetime import datetime
from math import ceil, sqrt
from datasets import utils

from NLastPeriods.Model import Model as NLastPeriodsModel
from SEAIPPF.Model import Model as SEAIPPFModel
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from SEAIPPF.Transformers.TransformerSimpleFiltering import TransformerSimpleFiltering
from sktimeSEAPF.Optimized import Optimized


MOVING_AVG_WINDOW_SIZE = 4
START_DATE = '2023-05-08'
END_DATE = '2023-05-10'
ONE_YEAR = 0, 288*360

DICTIONARY = {
    "en":{
        "Power": "Power",
        "Time": "Time",
        "Phase": "Phase",
        "Power [kW]": "Power [kW]",
        "Normalized power": "Normalized power",
        "Voltage [V]": "Voltage [V]",
        "Installation power": "Installation power",
        "plot_power_title": "PV power and PV normalized power for selected installations",
        "plot_voltage_title": "PV power and Grid Voltage",
        "plot_potential_losses": "Produced energy vs. production lost",
        "Grid overvoltage": "Grid overvoltage",
        "OV Threshold": "Overvoltage Threshold",
        "prone_installation_OV_vs_power": "PV power and Grid Voltage for prone installation",
        "healthy_installation_OV_vs_power": "PV power and Grid Voltage for healthy installation",
        "model_vs_power": "PV model prediction and PV actual power",
        "model_vs_looses": "PV model prediction and PV looses",
        "model_vs_looses_metric": "PV looses and PV model prediction quality metric meaning",
        "y_adjustment": "Y-adjustment on Linear Regression estimation",
        "model_representation": "Model representation on overlapped observation\nin solar elevation angle domain"
    },
    "pl":{
        "Power": "Moc",
        "Normalized power": "Moc znormalizowana",
    }

}

TEXTS = DICTIONARY["en"]

load_data_cache = None

def load_data():
    global load_data_cache
    if load_data_cache is not None:
        return load_data_cache["data"], load_data_cache["full_data"]

    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    load_data_cache = {
        "full_data": pd.read_csv(file_path, low_memory=False),
        "data": None
    }

    # self.full_data = self.full_data[30:]
    load_data_cache["full_data"]['timestamp'] = pd.to_datetime(load_data_cache["full_data"]['timestamp'])
    load_data_cache["full_data"].index = load_data_cache["full_data"]['timestamp']
    load_data_cache["full_data"].drop(columns=["timestamp"], inplace=True)

    load_data_cache["data"] = load_data_cache["full_data"].loc[START_DATE:END_DATE]
    return load_data_cache["data"], load_data_cache["full_data"]

class Analysis():
    def __init__(self):
        self.data, self.full_data = load_data()

        # self.data.fillna(0, inplace=True)

        # list = self.data.index.to_list()
        # for l in list:
        #     print(l)

        self.cache = {}

    @staticmethod
    def get_column_names(txt):
        return [f"{i}_{txt}" for i in range(0,4)]

    @staticmethod
    def get_pv_names():
        return [f"PV {i}" for i in range(0+1, 4+1)]

    @staticmethod
    def get_elevation(ts, latitude_degrees, longitude_degrees):
        """
        ts = df.index
        """
        timestamps = (ts.astype(int) // 10 ** 9).astype(int)
        elevation = Solar.elevation(Optimized.from_timestamps(timestamps), latitude_degrees,
                                    longitude_degrees) * 180 / np.pi
        return elevation

    def get_looses_curve(self, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, data=None):
        if data is None:
            data = self.data
        # if "get_looses_curve" in self.cache:
        #     return self.cache["get_looses_curve"]

        voltage_columns = [f"{prone_installation_id}_L1V", f"{prone_installation_id}_L2V",
                           f"{prone_installation_id}_L3V"]
        mx = data[voltage_columns].agg("max", axis="columns")
        mx = np.where(mx >= limit_voltage, 1, 0)
        mi = np.where(mx >= limit_voltage, 0, 1)

        df = data.loc[:, [f"{healthy_installation_id}_NormalizedPower"]]
        df.loc[:, "NormalizedPower"] = df[f"{healthy_installation_id}_NormalizedPower"] * mi
        df.insert(0, "NormalizedLooses", df[f"{healthy_installation_id}_NormalizedPower"] * mx)
        normalized_losses = df["NormalizedLooses"].copy()
        normalized_losses.loc[df["NormalizedLooses"] == 0] = np.nan
        normalized_power = df["NormalizedPower"].copy()
        normalized_power.loc[df["NormalizedLooses"] != 0] = 0

        # self.cache["get_looses_curve"] = pd.DataFrame({"NormalizedLooses": normalized_losses, "NormalizedPower": normalized_power}, index=df.index)
        # return self.cache["get_looses_curve"]
        return pd.DataFrame({"NormalizedLooses": normalized_losses, "NormalizedPower": normalized_power}, index=df.index)

    def plot_first_data(self):
        fig, ax = plt.subplots(1)
        df = self.full_data[0:2*288]

        elevation = Analysis.get_elevation(df.index,self.full_data["0_Latitude"][0],self.full_data["0_Longitude"][0])
        lines = []

        for name, pv_name in zip(Analysis.get_column_names("NormalizedPower"), Analysis.get_pv_names()):
            d = Optimized.window_moving_avg(df[[name]].to_numpy(), window_size=MOVING_AVG_WINDOW_SIZE, roll=True)
            l, = ax.plot(df.index, d, label=pv_name)
            lines.append(l)
        tax = ax.twinx()
        tax.plot(df.index, elevation, label="elevation", c="purple")
        ax.plot(df.index, np.zeros(len(df)), label="_zeros", c="black")

        fig.tight_layout()
        return fig

    def get_nlastperiods_model(self, installation_id=None, y_fit=None, model_parameters={}, fit_model=True):
        model = NLastPeriodsModel(N=model_parameters.get("N", 3))

        if fit_model: #fit only if it is needed (if fot data is defined)
            model.fit(y=y_fit)

        return model

    def get_seaippf_model(self, installation_id=None, y_fit=None, model_parameters={}, fit_model=True):
        # if not (y_fit is None or installation_id is None):
        #     y_fit = self.full_data[f"{installation_id}_NormalizedPower"]

        bins = model_parameters["bins"] if "bins" in model_parameters else 30
        #0.21 MAE
        # model = SEAIPPFModel(
        #     latitude_degrees=self.full_data[f"{installation_id}_Latitude"][0],
        #     longitude_degrees=self.full_data[f"{installation_id}_Longitude"][0],
        #     x_bins=bins,
        #     y_bins=bins,
        #     bandwidth=model_parameters.get("bandwidth", 0.05),
        #     interpolation=True,
        #     enable_debug_params= True,
        #     transformer=TransformerTest(
        #         regressor_degrees=model_parameters.get("regressor_degrees", 12),
        #         hit_points_max_iter=model_parameters.get("hit_points_max_iter", 1),
        #         hit_points_neighbourhood=int(model_parameters.get("hit_points_neighbourhood", 15) * bins),
        #         conv2Dx_shape_factor = model_parameters.get("conv2Dx_shape_factor", 0.05),
        #         kde_r = model_parameters.get("kde_r", 0.05),
        #     )
        # )

        model = SEAIPPFModel(
            latitude_degrees=self.full_data[f"{installation_id}_Latitude"][0],
            longitude_degrees=self.full_data[f"{installation_id}_Longitude"][0],
            x_bins=bins,
            y_bins=bins,
            bandwidth=model_parameters.get("bandwidth", 0.05),
            interpolation=True,
            enable_debug_params=True,
            transformer=TransformerTest(
                regressor_degrees=model_parameters.get("regressor_degrees", 12),
                iterative_regressor_degrees=model_parameters.get("iterative_regressor_degrees", 12),
                conv2D_shape_factor=model_parameters.get("conv2D_shape_factor", 0.05),
                iterations=model_parameters.get("iterations", 3),
                stretch=model_parameters.get("stretch", True)
                # regressor_degrees=model_parameters.get("regressor_degrees", 12),
                # hit_points_max_iter=model_parameters.get("hit_points_max_iter", 1),
                # hit_points_neighbourhood=model_parameters.get("hit_points_neighbourhood", 0.1),
                # conv2Dx_shape_factor=model_parameters.get("conv2Dx_shape_factor", 0.05),
                # kde_r=model_parameters.get("kde_r", 0.05),
                # kde_epsilon=model_parameters.get("kde_epsilon", 1/4),
            )
        )

        # model.fit(y=self.full_data[f"{healthy_installation_id}_NormalizedPower"])

        if fit_model: #fit only if it is needed (if fot data is defined)
            # y_fit = y_fit[y_fit.first_valid_index():]
            model.fit(y=y_fit[~np.isnan(y_fit)])
        return model

    def get_experiment_dataset(self,prone_installation_id=3, healthy_installation_id=0, limit_voltage=256,
                               window_size=288, require_minimum_samples_count=6, split_data=True,
                               validation_fraction=4, validation=False):
        df = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage,
                                   data=self.full_data)
        DAY_LEN = 24 * 60 * 60
        df["timestamp"] = df.index.astype('datetime64[s]').astype('int')
        df["Bin"] = (df["timestamp"] - df["timestamp"].min()) // DAY_LEN
        df["Incremental_id"] = range(len(df))
        # df["FirstTimestamp"] = df.groupby("Bin").min()["timestamp"]
        other_groups = df[["timestamp", "Bin", "Incremental_id"]].groupby("Bin")
        other = pd.DataFrame({
            "timestamp_by_bin": other_groups["timestamp"].min(),
            "Incremental_id_begin": other_groups["Incremental_id"].min(),
            "Incremental_id_end": other_groups["Incremental_id"].max()
        })
        df = df.join(other, on="Bin")
        df['timestamp_by_bin'] = pd.to_datetime(df['timestamp_by_bin'], unit="s")

        # limit data to OV days
        ov_happened_days_group = df.loc[~np.isnan(df["NormalizedLooses"]),
                                  ["Bin", "Incremental_id_begin", "Incremental_id_end"]].groupby("Bin")
        ov_happened_days = pd.DataFrame({
            "Train End" : ov_happened_days_group["Incremental_id_begin"].min(),
            "Test End" : ov_happened_days_group["Incremental_id_end"].min(),
            "Count" : ov_happened_days_group.size(),
        })
        ov_happened_days = ov_happened_days[ov_happened_days["Count"] > require_minimum_samples_count] #use only those days during OV last more than 6 samples
        ov_happened_days["Test End"] = ov_happened_days["Test End"] + 1  # index range <n,m) <- right opened
        ov_happened_days["Test Begin"] = ov_happened_days["Train End"]
        ov_happened_days["Train Begin"] = ov_happened_days["Train End"] - window_size
        # print("List of days during grid OV happened\n", ov_happened_days)

        if split_data:
            y_test, y_train, y_true = [], [], []
            for i, row in ov_happened_days.iterrows():
                y_train.append(df.iloc[row["Train Begin"]:row["Train End"]]["NormalizedPower"])
                y_test.append(df.iloc[row["Test Begin"]:row["Test End"]]["NormalizedPower"])
                y_true.append(df.iloc[row["Test Begin"]:row["Test End"]]["NormalizedLooses"])

            indices = list(range(len(y_test)))
            if not validation:
                indices = indices[::validation_fraction]
            else:
                indices = [i for i in indices if not i in indices[::validation_fraction]]

            y_test, y_train, y_true = [y_test[i] for i in indices],\
                                      [y_train[i] for i in indices],\
                                      [y_true[i] for i in indices]

            return y_test, y_train, y_true
        else:
            return df["NormalizedPower"], df["NormalizedLooses"]

    def plot_power(self):
        fig, ax = plt.subplots(2)
        df = self.data

        fig.suptitle(TEXTS["plot_power_title"])
        _ax = ax[0]
        lines = []
        for name, pv_name in zip(Analysis.get_column_names("Power"), Analysis.get_pv_names()):
            d = Optimized.window_moving_avg(df[[name]].to_numpy(), window_size=MOVING_AVG_WINDOW_SIZE, roll=True)
            l, = _ax.plot(df.index, d, label=pv_name)
            lines.append(l)
        _ax.plot(df.index, np.zeros(len(df)), label="_zeros", c="black")
        _ax = ax[1]

        for name in Analysis.get_column_names("NormalizedPower"):
            d = Optimized.window_moving_avg(df[[name]].to_numpy(), window_size=MOVING_AVG_WINDOW_SIZE, roll=True)
            _ax.plot(df.index, d)
        _ax.plot(df.index, np.zeros(len(df)), label="_zeros", c="black")

        ax[0].set_xlabel("")
        ax[0].set_ylabel(TEXTS["Power [kW]"])
        ax[1].set_xlabel(TEXTS["Time"])
        ax[1].set_ylabel(TEXTS["Normalized power"])
        ax[0].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax[0].legend(lines, [l.get_label() for l in lines])

        fig.tight_layout()
        return fig

    def plot_voltage(self, installation_id=3, limit_voltage = 256, title=None):
        fig, ax = plt.subplots(2)
        voltage_columns = [f"{installation_id}_L1V", f"{installation_id}_L2V", f"{installation_id}_L3V"]
        df = self.data[[f"{installation_id}_NormalizedPower"] + voltage_columns]
        _ax = ax[0]
        lines = []
        for i, name in enumerate(voltage_columns):
            d = df[[name]].to_numpy()
            d[d<200] = np.nan
            l, = _ax.plot(df.index, d, label=f"{TEXTS['Phase']} {i+1}")
            lines.append(l)

        l, = _ax.plot(df.index, [limit_voltage] * len(df.index), c="red", label=TEXTS["OV Threshold"])
        lines.append(l)

        mx = self.data[voltage_columns].agg("max", axis="columns")
        mx = np.where(mx >= limit_voltage, 1, 0)

        boundaries = []
        is_in_peak = False
        for i, _mx in enumerate(mx):
            if not is_in_peak and _mx:
                is_in_peak = True
                boundaries.append([i])
            if not _mx and is_in_peak:
                is_in_peak = False
                boundaries[-1].append(i)
        l = None
        for b in boundaries:
            ax[0].axvspan(df.index[b[0]], df.index[b[1]], color="gray", alpha=0.2, label=TEXTS["Grid overvoltage"])
            l = ax[1].axvspan(df.index[b[0]], df.index[b[1]], color="gray", alpha=0.2, label=TEXTS["Grid overvoltage"])
        if l is not None:
            lines.append(l)

        _ax = ax[1]
        d = df[[f"{installation_id}_NormalizedPower"]].to_numpy()
        l, = _ax.plot(df.index, d, label=TEXTS["Installation power"])
        _ax.plot(df.index, np.zeros(len(df)), label="_zeros", c="black")
        lines.append(l)

        ax[0].legend(lines, [l.get_label() for l in lines])

        fig.suptitle(TEXTS["plot_voltage_title"] if title is None else title)
        ax[0].set_ylabel(TEXTS["Voltage [V]"])
        ax[1].set_xlabel(TEXTS["Time"])
        ax[1].set_ylabel(TEXTS["Normalized power"])
        ax[0].set_ylim([220, 260])
        ax[0].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

        fig.tight_layout()
        return fig

    def plot_potential_losses(self, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, bg_alpha=0.3):

        df = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage)

        fig, ax = plt.subplots(1)

        _ax = ax
        lines = []

        # l, = _ax.plot(df.index, data, linestyle='dashed', marker='+', label=f"Looses (Energy not produced)")
        l, = _ax.plot(df.index, df["NormalizedLooses"], label=f"Looses (Power not generated) - ground true", c="black", linestyle="dashed")
        lines.append(l)

        l, = _ax.plot(df.index, df["NormalizedPower"], label=f"Power (Power generated)", c="blue")
        lines.append(l)
        zeros = np.zeros(len(df))

        l = _ax.fill_between(x=df.index, y1=df["NormalizedPower"], color="blue", alpha=bg_alpha, label = "Energy Produced")
        lines.append(l)
        l = _ax.fill_between(x=df.index, y1=df["NormalizedLooses"], color="black", alpha=bg_alpha, label = "Energy Lost")
        lines.append(l)

        ax.plot(df.index, zeros, label="_zeros", c="black")
        fig.suptitle(TEXTS["plot_potential_losses"])
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax.set_ylabel(TEXTS["Normalized power"])
        ax.set_xlabel(TEXTS["Time"])
        ax.legend(lines, [l.get_label() for l in lines])

        fig.tight_layout()
        return fig

    def plot_seaippf_model_parameters(self, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, model_parameters={}, config=1):
        # if model_factory is None:
        #     raise ValueError("model_factory cannot be none!")
        data = self.full_data
        if config == 0:
            data = self.full_data[self.full_data.first_valid_index():]
            data = data[ONE_YEAR[0]:3*ONE_YEAR[1]//5]
        fit_data = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage,
                                         data=data)

        # print("len(fit_data)", len(fit_data))
        model = self.get_seaippf_model(
            healthy_installation_id,
            y_fit=fit_data["NormalizedPower"],
            model_parameters=model_parameters)

        z = model_parameters
        z["stretch"] = True
        model_stretch = self.get_seaippf_model(
            healthy_installation_id,
            y_fit=fit_data["NormalizedPower"],
            model_parameters=z)



        # fig1, ax1 = model.transformer.plot()
        # fig1.tight_layout()
        fig2, ax2 = model.plot()
        fig2.tight_layout()
        # model.plot(plots=["y_adjustment"])
        fig3, ax3 = model.plot_model()
        ax3.plot(model_stretch.plot_model_x_axis(), model_stretch.model_representation_, color="g", label="Proposed model representation")
        ax3.legend()
        # fig3.suptitle(TEXTS["model_representation"] + f"for dataset {healthy_installation_id}")
        fig4, ax4 = model.plot_y_adjustment()
        fig4.suptitle(TEXTS["y_adjustment"] + f"for dataset {healthy_installation_id}")

        return [
            # fig1,
            fig2,
            fig3,
            fig4]

    def plot_model_vs_power(self, model_factory, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, model_parameters={}):
        if model_factory is None:
            raise ValueError("model_factory cannot be none!")

        df = self.data
        fit_data = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage,data=self.full_data)
        power_estimation = model_factory(
            self,
            healthy_installation_id,
            y_fit=fit_data["NormalizedPower"],
            model_parameters=model_parameters).in_sample_predict(X=None, ts=df.index)

        fig, ax = plt.subplots(1)
        fig.suptitle(TEXTS["model_vs_power"])
        _ax = ax
        lines = []
        l, = _ax.plot(df.index, df[f"{healthy_installation_id}_NormalizedPower"], label=f"Power - ground true", c="black", linestyle="dashed")
        lines.append(l)

        l, = _ax.plot(df.index, power_estimation, label=f"Power - prediction", c="orange")
        lines.append(l)

        # elevation = Analysis.get_elevation(df.index, self.full_data["0_Latitude"][0], self.full_data["0_Longitude"][0])
        # l, = _ax.plot(df.index, elevation, label=f"Elevation_", c="red")

        _ax.legend(lines, [l.get_label() for l in lines])
        _ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        _ax.set_ylabel(TEXTS["Normalized power"])
        _ax.set_xlabel(TEXTS["Time"])

        fig.tight_layout()
        return fig

    def plot_model_vs_looses(self, model_factory, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, model_parameters={}):
        if model_factory is None:
            raise ValueError("model_factory cannot be none!")

        df = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage)
        df.loc[~np.isnan(df["NormalizedLooses"]), "NormalizedPower"] = np.nan

        fit_data = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage, data=self.full_data)
        power_estimation = model_factory(
            self,
            healthy_installation_id,
            y_fit=fit_data["NormalizedPower"],
            model_parameters=model_parameters).in_sample_predict(X=None, ts=df.index)
        power_estimation[np.isnan(df["NormalizedLooses"])] = np.nan

        fig, ax = plt.subplots(1)
        fig.suptitle(TEXTS["model_vs_looses"])
        _ax = ax
        lines = []

        l, = _ax.plot(df.index, df["NormalizedPower"], label=f"Power", c="blue")
        lines.append(l)

        l, = _ax.plot(df.index, df["NormalizedLooses"], label=f"Looses (Power not generated) - ground true", c="black", linestyle="dashed")
        lines.append(l)

        l, = _ax.plot(df.index, power_estimation, label=f"Looses (Power not generated) - prediction", c="orange")
        lines.append(l)

        ax.legend(lines, [l.get_label() for l in lines])
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax.set_ylabel(TEXTS["Normalized power"])
        ax.set_xlabel(TEXTS["Time"])

        fig.tight_layout()
        return fig

    def plot_model_vs_looses_metric(self, model_factory, prone_installation_id=3, healthy_installation_id=0, limit_voltage = 256, model_parameters={}):
        if model_factory is None:
            raise ValueError("model_factory cannot be none!")

        df = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage)
        df.loc[~np.isnan(df["NormalizedLooses"]), "NormalizedPower"] = np.nan

        fit_data = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage,data=self.full_data)
        power_estimation = model_factory(
            self,
            healthy_installation_id,
            y_fit=fit_data["NormalizedPower"],
            model_parameters=model_parameters).in_sample_predict(X=None,ts=df.index)
        power_estimation[np.isnan(df["NormalizedLooses"])] = np.nan

        fig, ax = plt.subplots(1)
        fig.suptitle(TEXTS["model_vs_looses_metric"])
        _ax = ax
        lines = []

        l, = _ax.plot(df.index, df["NormalizedLooses"], label=f"Looses (Power not generated) - ground true", c="black", linestyle="dashed")
        lines.append(l)

        l, = _ax.plot(df.index, power_estimation, label=f"Looses (Power not generated) - prediction", c="orange")
        lines.append(l)

        l = _ax.fill_between(df.index, df["NormalizedLooses"], power_estimation, label="Metric meaning")
        lines.append(l)

        ax.legend(lines, [l.get_label() for l in lines])
        ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        ax.set_ylabel(TEXTS["Normalized power"])
        ax.set_xlabel(TEXTS["Time"])

        fig.tight_layout()
        return fig

    def prone_healthy_classification(self, installation_ids=0, limit_voltage = 256, command_name="proneHealthyClassification"):
        dictionary = {
            "Prone factor": [],
            "OV": [],
            "Length": [],
            "Begin": [],
            "End": [],
        }

        for id in installation_ids:
            d = self.full_data[f"{id}_NormalizedPower"]

            voltage_columns = [f"{id}_L1V", f"{id}_L2V",
                               f"{id}_L3V"]
            data = self.full_data.loc[d.first_valid_index():,voltage_columns]
            mx = data.agg("max", axis="columns")
            mx = np.where(mx >= limit_voltage , 1, 0)
            dictionary["Prone factor"].append(f"{100*sum(mx) / len(mx):.0f}\%")
            dictionary["OV"].append(sum(mx))
            dictionary["Length"].append(len(mx))
            dictionary["Begin"].append(str(data.index[0].strftime("%d-%m-%Y")))
            dictionary["End"].append(str(data.index[-1].strftime("%d-%m-%Y")))

        txt = "\\newcommand*\\" + command_name +"{" + \
            pd.DataFrame(dictionary, index=pd.Index([f"PV {i+1}" for i in installation_ids], name='Id')).to_latex() + \
        "}"

        return txt

    def plot(self):
        # plotter = Plotter(self.data.index, [self.data[[i]] for i in ["1_Power", "1_L1V", "1_L2V", "1_L3V"]], debug=False)
        plotter = Plotter(self.data.index, [self.data[[f"{i}_Power"]] for i in range(0,4)], debug=True)
        plotter.show()


    def simulation(self, model_factory, prone_installation_id=3, healthy_installation_ids=[0,1], limit_voltage=256):
        if model_factory is None:
            raise ValueError("model_factory cannot be none!")

        predictions = []
        looses = []
        metrics = [
            ("MAE", mean_absolute_error),
            ("MSE", mean_squared_error),
            ("TE", lambda g,p: 1- (mean_absolute_error(g,p) / abs(g).mean())),
            ("Count", lambda g,p: len(g)),
        ]
        metric_results = []
        for healthy_installation_id in healthy_installation_ids:
            fit_data = self.get_looses_curve(prone_installation_id, healthy_installation_id, limit_voltage, data=self.full_data, model_parameters={})
            model = model_factory(
                self,
                healthy_installation_id,
                y_fit=fit_data["NormalizedPower"],
                model_parameters=model_parameters)
            ground_true = fit_data.loc[~np.isnan(fit_data["NormalizedLooses"]), "NormalizedLooses"]

            power_estimation = model.in_sample_predict(X=None, ts=ground_true.index)
            ground_true = ground_true.to_numpy()
            # print(power_estimation)
            pred = power_estimation.fillna(0).to_numpy()

            mr = []
            for metric in metrics:
                mr.append(metric[1](ground_true, pred))
            metric_results.append(mr)

            # predictions.append(prediction)
            # looses.append(lose)

        data = {m[0]: [metric_results[i][j] for i in range(len(healthy_installation_ids))] for j,m in enumerate(metrics)}

        results = pd.DataFrame(data, columns=[m[0] for m in metrics], index=[f"{i}_installation" for i in healthy_installation_ids])
        print(results)
        return results

    def timeseries_evaluation(self, model_factory, installation_id, model_parameters={}, **kwargs):
        if model_factory is None:
            raise ValueError("model_factory cannot be none!")

        data = self.full_data[f"{installation_id}_NormalizedPower"]
        data = data[data.first_valid_index():]

        # result, evaluate = Evaluate.scipy_differential_evolution(
        #     data = series.to_numpy(),
        #     ts = series.index.astype(int) // 10e8,
        #     model=model_factory(self, healthy_installation_id),
        #     include_parameters=include_parameters)

        experiment = Experimental()

        # register timeseries
        experiment.register_dataset(data)

        # register models
        if type(model_factory) is list:
            experiment.register_models([
                mf(self, installation_id, fit_model=False, model_parameters=mp) for mf, mp in zip(model_factory, model_parameters)
            ])
        else:
            experiment.register_models([
                model_factory(self, installation_id, fit_model=False, model_parameters=model_parameters)
            ])

        # register metrics and set theirs names (used int returned pandas dataframe)
        experiment.register_metrics([
            (mean_absolute_error, "MAE"),
            (mean_squared_error, "MSE"),
            (r2_score, "R2"),
        ])

        # make experiment
        predictions, metrics_results, forecast_start_point = experiment.predict(
            forecast_horizon=kwargs.get('forecast_horizon', 288),
            batch=kwargs.get('batch', 288),
            window_length=kwargs.get('window_length', 288 * 100),
            early_stop=kwargs.get('early_stop', 288 * 2), # define early stop - before all data are used
        )

        return predictions, metrics_results

    def plot_model_test_dataset(self,prone_installation_id=3, healthy_installation_id=0, limit_voltage=256, window_size=2*288, bg_alpha=0.4):
        y_test, y_train, y_true = self.get_experiment_dataset(prone_installation_id, healthy_installation_id, limit_voltage, window_size)
        print("len:", len(y_test), len(y_train), len(y_true))

        fig, ax = plt.subplots(1)
        # fig.suptitle(TEXTS["model_vs_looses_metric"])
        _ax = ax
        lines = []

        # index =  267
        index =  0

        for i, val in enumerate(y_true):
            if val.index[0] > pd.Timestamp('2023-05-08'):
                index = i
                break

        # END_DATE = '2023-05-10'
        msk = np.isnan(y_true[index])
        y_test[index].iloc[~msk] = np.nan # only for display
        y_true[index].iloc[msk] = y_test[index][msk] # only for display

        metric_calc_period = y_true[index].iloc[:]
        metric_calc_period[msk] = np.nan

        _ax.plot(y_true[index].index, y_true[index], label="Ground true",  c="black", linestyle="dashed")
        _ax.plot(y_test[index].index, y_test[index], label="Test set", c="blue")
        _ax.plot(y_train[index].index, y_train[index], label="Train set", c="orange")
        l = _ax.fill_between(x=y_true[index].index, y1=metric_calc_period, color="blue", alpha=bg_alpha,
                             label="Metric calculation period")

        _ax.legend()
        return fig

