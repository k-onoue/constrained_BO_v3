import argparse
import logging
import os
from functools import partial

import random 

import numpy as np
import optuna
from _src import DB_DIR, LOG_DIR, WarcraftObjective, set_logger
from optuna.samplers import BruteForceSampler, RandomSampler, TPESampler
from optuna.samplers import CustomGPSampler as GPSampler
from optuna.samplers import GPSampler as HGPSampler


def sphere(x):
    return np.sum(x ** 2)


def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum1 + sum2 + a + np.exp(1)


def objective(trial, dimension=None, function=None, map_shape=None, objective_function=None):
    if function in ["sphere", "ackley"]:
        categories = list(range(-5, 6))

        x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(dimension)])
        if function == "sphere":
            return sphere(x)
        elif function == "ackley":
            return ackley(x)
    elif function == "warcraft":
        directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]

        x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):

                x[i, j] = trial.suggest_categorical(f"x_{i}_{j}", directions)
        return objective_function(x)
    else:
        raise ValueError(f"Unsupported function type: {function}")


def run_bo(settings):
    random.seed(settings['seed'])

    function = settings["function"]
    
    if function in ["sphere", "ackley"]:
        dimension = settings["dimension"]
        objective_with_args = partial(objective, dimension=dimension, function=function)
    elif function == "warcraft":
        map_targeted = settings["map"]
        map_shape = map_targeted.shape
        objective_function = WarcraftObjective(map_targeted)
        objective_with_args = partial(objective, map_shape=map_shape, objective_function=objective_function, function=function)
    else:
        raise ValueError(f"Unsupported function type: {function}")

    sampler = settings["sampler"]
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"

    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
    )
    logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")

    if function == "warcraft":
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        logging.info(f"Best Direction Matrix:\n{best_x}")

    optuna.visualization.plot_optimization_history(study)


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Optimization Benchmark Experiment")
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of iterations for Bayesian optimization.")
    parser.add_argument("--acq_trade_off_param", type=float, default=3.0, help="Trade-off parameter for the acquisition function.")
    parser.add_argument("--acq_batch_size", type=int, default=10, help="Batch size for optimization.")
    parser.add_argument("--acq_maximize", action="store_true", help="Whether to maximize the acquisition function.")
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "hgp", "gp", "bruteforce"], default="random", help="Sampler for the optimization process.")
    parser.add_argument("--n_startup_trials", type=int, default=10, help="Number of initial trials for the optimization process.")
    parser.add_argument("--dimension", type=int, default=2, help="Number of dimensions for the function.")
    parser.add_argument("--function", type=str, choices=["sphere", "ackley", "warcraft"], default="sphere", help="Objective function to optimize.")
    parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1, help="Select the map configuration: 1 for 2x2, 2 for 3x2, 3 for 3x3 (only for Warcraft).")
    return parser.parse_args()


def get_sampler(sampler_type: str, seed: int):
    if sampler_type == "random":
        return RandomSampler(seed=seed)
    elif sampler_type == "tpe":
        return TPESampler(seed=seed, n_startup_trials=args.n_startup_trials)
    elif sampler_type == "hgp":
        return HGPSampler(seed=seed, n_startup_trials=args.n_startup_trials)
    elif sampler_type == "gp":
        return GPSampler(seed=seed, n_startup_trials=args.n_startup_trials)
    elif sampler_type == "bruteforce":
        return BruteForceSampler()
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def get_map(map_option: int):
    if map_option == 1:
        map_targeted = np.array([[1, 4], [2, 1]])
    elif map_option == 2:
        map_targeted = np.array([[1, 4, 1], [2, 1, 1]])
    elif map_option == 3:
        map_targeted = np.array([[1, 4, 1], [2, 1, 3], [5, 2, 1]])
    else:
        raise ValueError(f"Invalid map option: {map_option}")
    return map_targeted / map_targeted.sum()


if __name__ == "__main__":
    base_script_name = os.path.splitext(__file__.split("/")[-1])[0]
    args = parse_args()

    timestamp = args.timestamp
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    if args.function == "warcraft":
        map_targeted = get_map(args.map_option)
        log_filename_base = f"{args.function}_map{args.map_option}_{args.sampler}_seed{args.seed}"
    else:
        map_targeted = None
        log_filename_base = f"{args.function}_dim{args.dimension}_{args.sampler}_seed{args.seed}"

    log_filepath = set_logger(log_filename_base, results_dir)

    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    sampler = get_sampler(args.sampler, args.seed)

    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "dimension": args.dimension,
        "function": args.function,
        "map": map_targeted,
        "iter_bo": args.iter_bo,
        "storage": storage_url,
        "sampler": sampler,
        "acqf_settings": {
            "trade_off_param": args.acq_trade_off_param,
            "batch_size": args.acq_batch_size,
            "maximize": args.acq_maximize,
        },
    }

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)