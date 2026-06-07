from Analysis import Analysis


def lastNPeriods_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "N": [1, 30, True],
        "window_size_days_10": [4, 4, True],
    })


def seaippf_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "Bins": [20, 40, True],
        "window_size_days_10": [20, 40, True],  # at 31 elbow
        "conv2D_shape_factor": [0, 0.05, False],
        "regressor_degrees": [8, 15, True],
        "bandwidth": [0.01, 0.1, False],
        "iterations": [1, 6, True],
        "iterative_regressor_degrees": [10, 15, True],
        "stretch": [1, 1, True],
    })


def lr_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [2, 30, True],
        "degree": [5, 17, True],
    })


def rf_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [2, 30, True],
        "n_estimators*5": [1, 20, True],
    })


def mlp_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [2, 30, True],
        "l1*5": [1, 8, True],
        "l2*5": [1, 8, True],
        "l3*5": [1, 8, True],
    })


def complex_mlp_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [2, 30, True],
        "n": [1, 10, True],
        "n_step": [1, 10, True],
    })


def lstm_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "window_size_days_10": [2, 30, True],
    })


def cnn_model(simulation, args):
    simulation.generic_evaluate(parameters={
        "complexity2^x": [1, 6, True],
        "n": [2, 10, True],
        "n_step": [1, 10, True],
    })


MODEL_CONFIGS = {
    "seaippf": (Analysis.get_seaippf_model, seaippf_model, {"save_each_image": 1}),
    "nlastperiods": (Analysis.get_nlastperiods_model, lastNPeriods_model, {}),
    "lr": (Analysis.get_lr_model, lr_model, {}),
    "rf": (Analysis.get_rf_model, rf_model, {}),
    "mlp": (Analysis.get_mlp_model, mlp_model, {}),
    "lstm": (Analysis.get_lstm_model, lstm_model, {}),
    "cnn": (Analysis.get_cnn_model, cnn_model, {}),
    "complex_mlp": (Analysis.get_complex_mlp_model, complex_mlp_model, {}),
}

VALIDATION_MODEL_CONFIGS = {
    model: (model_factory, extra_simulation_kwargs)
    for model, (model_factory, _evaluate_model, extra_simulation_kwargs) in MODEL_CONFIGS.items()
}

MODEL_CHOICES = list(MODEL_CONFIGS.keys())
MODEL_NAMES_BY_LENGTH = sorted(MODEL_CHOICES, key=len, reverse=True)


def get_model_from_dirname(dirname):

    if dirname.endswith("_validation"):
        dirname = dirname[:-len("_validation")]

    for model in MODEL_NAMES_BY_LENGTH:
        if dirname.endswith(f"_{model}"):
            return model

    return None
