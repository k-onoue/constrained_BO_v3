import argparse
import logging
import os
from functools import partial

import random
import numpy as np
import optuna
from _src import TFSampler, WarcraftObjective, set_logger


def sphere(x):
    """
    Computes the d-dimensional Sphere function.
    :param x: np.array, shape (d,) - point at which to evaluate the function.
    :return: float - value of the Sphere function at x.
    """
    return np.sum(x ** 2)


def ackley(x):
    """
    Computes the d-dimensional Ackley function.
    :param x: np.array, shape (d,) - point at which to evaluate the function.
    :return: float - value of the Ackley function at x.
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    sum2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum1 + sum2 + a + np.exp(1)


def objective(trial, dimension=None, function=None, map_shape=None, objective_function=None):
    """
    Objective function for Bayesian optimization.
    """
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

    sampler = TFSampler(
        seed=settings["seed"],
        method=settings["tf_settings"]["method"],
        acquisition_function=settings["acqf_settings"]["acquisition_function"],
        sampler_params={
            "n_startup_trials": settings["sampler_settings"].get("n_startup_trials", 1),
            "decomp_iter_num": settings["sampler_settings"].get("decomp_iter_num", 10),
            "mask_ratio": settings["sampler_settings"].get("mask_ratio", 0.9),
            "include_observed_points": settings["sampler_settings"].get("include_observed_points", False),
            "unique_sampling": settings["sampler_settings"].get("unique_sampling", False),
        },
        tf_params={
            "rank": settings["tf_settings"]["rank"],
            "lr": settings["tf_settings"]["optim_params"]["lr"],
            "max_iter": settings["tf_settings"]["optim_params"]["max_iter"],
            "tol": settings["tf_settings"]["optim_params"]["tol"],
            "reg_lambda": settings["tf_settings"]["optim_params"]["reg_lambda"],
            "constraint_lambda": settings["tf_settings"]["optim_params"]["constraint_lambda"],
            "fill_constraint_method": settings["tf_settings"]["optim_params"]["fill_constraint_method"],
        },
        acqf_params={
            "trade_off_param": settings["acqf_settings"]["trade_off_param"],
        }
    )

    direction = "maximize" if settings["direction"] else "minimize"

    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
    )

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    if function == "warcraft":
        best_x = np.empty(map_shape, dtype=object)
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                best_x[i, j] = study.best_params[f"x_{i}_{j}"]
        print(f"Best Direction Matrix:\n{best_x}")

    optuna.visualization.plot_optimization_history(study)

def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian Optimization with Tensor Factorization")
    # Basic parameters
    parser.add_argument("--timestamp", type=str, help="Timestamp for the experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iter_bo", type=int, default=300, help="Number of BO iterations")
    parser.add_argument("--dimension", type=int, default=2, help="Problem dimension")
    parser.add_argument("--function", type=str, choices=["sphere", "ackley", "warcraft"], default="sphere")
    parser.add_argument("--map_option", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--direction", action="store_true", help="Maximize the objective function")

    # TF-specific arguments
    parser.add_argument("--tf_method", type=str, choices=["cp", "tucker", "train", "ring"], default="cp")
    parser.add_argument("--tf_rank", type=int, default=3, help="Tensor rank")
    parser.add_argument("--tf_lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--tf_max_iter", type=int, default=1000, help="Max iterations")
    parser.add_argument("--tf_tol", type=float, default=1e-5, help="Convergence tolerance")
    parser.add_argument("--tf_reg_lambda", type=float, default=1e-3, help="Regularization strength")
    parser.add_argument("--tf_constraint_lambda", type=float, default=1.0, help="Constraint penalty")
    parser.add_argument("--tf_fill_method", type=str, choices=["zero", "normal", "minmax"], default="zero")
    
    # Sampler parameters
    parser.add_argument("--decomp_iter_num", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser.add_argument("--include_observed_points", action="store_true")
    parser.add_argument("--unique_sampling", action="store_true")
    parser.add_argument("--n_startup_trials", type=int, default=1)

    # Acquisition function arguments
    parser.add_argument("--acquisition_function", type=str, choices=["ucb", "ei"], default="ucb")
    parser.add_argument("--acq_trade_off_param", type=float, default=1.0)
    
    return parser.parse_args()


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
        log_filename_base = f"{args.function}_map{args.map_option}_{args.tf_method}_seed{args.seed}"
    else:
        map_targeted = None
        log_filename_base = f"{args.function}_dim{args.dimension}_{args.tf_method}_seed{args.seed}"

    log_filepath = set_logger(log_filename_base, results_dir)

    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    settings = {
        "name": f"{timestamp}_{log_filename_base}",
        "seed": args.seed,
        "dimension": args.dimension,
        "function": args.function,
        "direction": args.direction,
        "map": map_targeted,
        "iter_bo": args.iter_bo,
        "storage": storage_url,
        "tf_settings": {
            "method": args.tf_method,
            "rank": args.tf_rank,
            "optim_params": {
                "lr": args.tf_lr,
                "max_iter": args.tf_max_iter,
                "tol": args.tf_tol,
                "reg_lambda": args.tf_reg_lambda,
                "constraint_lambda": args.tf_constraint_lambda,
                "fill_constraint_method": args.tf_fill_method,
            }
        },
        "acqf_settings": {
            "acquisition_function": args.acquisition_function,
            "trade_off_param": args.acq_trade_off_param,
        },
        "sampler_settings": {
            "decomp_iter_num": args.decomp_iter_num,
            "mask_ratio": args.mask_ratio,
            "include_observed_points": args.include_observed_points,
            "unique_sampling": args.unique_sampling,
            "n_startup_trials": args.n_startup_trials,
        },
    }

    logging.info(f"Experiment settings: {settings}")
    run_bo(settings)