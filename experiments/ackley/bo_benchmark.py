import argparse
import logging
import os
from functools import partial

import numpy as np
import optuna
from _src import DB_DIR, LOG_DIR, set_logger, parse_experiment_path
from optuna.samplers import (BruteForceSampler, RandomSampler, TPESampler, GPSampler)


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


def objective(trial, dimensions):
    """
    Objective function for the Bayesian optimization using the d-dimensional Ackley function.
    This version restricts the suggested values to integers only.
    """
    # Suggest integer-valued parameters in each dimension within the range [-5, 5]
    x = np.array([trial.suggest_int(f"x_{i}", -5, 5) for i in range(dimensions)])
    
    # Compute the Ackley function
    return ackley(x)


def run_bo(settings):
    """
    Run the Bayesian optimization experiment using the specified settings.
    """
    dimensions = settings["dimensions"]  # Number of dimensions for the Ackley function
    
    # Set up the sampler for Bayesian optimization
    sampler = settings["sampler"]

    # Determine whether to minimize or maximize based on acq_maximize flag
    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"

    # Create or load the study
    try:
        study = optuna.load_study(
            study_name=settings["name"], storage=settings["storage"]
        )
        logging.info(f"Resuming study '{settings['name']}' from {settings['storage']}")
    except KeyError:
        study = optuna.create_study(
            study_name=settings["name"],
            sampler=sampler,
            direction=direction,
            storage=settings["storage"],
        )
        logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")

    # Bind the dimension to the objective function
    objective_with_args = partial(objective, dimensions=dimensions)

    # Optimize
    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best params: {study.best_params}")

    ###########################################################################
    # Plot the optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(title=f"Ackley: {settings['name']}")
    fig.show()



def parse_args():
    """
    Parse command-line arguments to specify experiment settings.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization Benchmark Experiment"
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--iter_bo",
        type=int,
        default=300,
        help="Number of iterations for Bayesian optimization (ignored for BruteforceSampler).",
    )
    parser.add_argument(
        "--acq_trade_off_param",
        type=float,
        default=3.0,
        help="Trade-off parameter for the acquisition function.",
    )
    parser.add_argument(
        "--acq_batch_size", type=int, default=10, help="Batch size for optimization."
    )
    parser.add_argument(
        "--acq_maximize",
        action="store_true",
        help="Whether to maximize the acquisition function.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["random", "tpe", "gp", "bruteforce"],
        default="random",
        help="Sampler for the optimization process.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=2,
        help="Number of dimensions for the Ackley function."
    )

    return parser.parse_args()


def get_sampler(sampler_type: str, seed: int):
    """
    Get the sampler instance based on the provided type.
    """
    if sampler_type == "random":
        return RandomSampler(seed=seed)
    elif sampler_type == "tpe":
        return TPESampler(seed=seed)
    elif sampler_type == "gp":
        return GPSampler(seed=seed)  # Use GPSampler for GP-based Bayesian Optimization
    elif sampler_type == "bruteforce":
        return BruteForceSampler()
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


if __name__ == "__main__":
    # Get the full path of the current script
    file_path = __file__

    # Parse the file path to get the base_script_name
    base_script_name = parse_experiment_path(file_path)

    # Parse the command-line arguments
    args = parse_args()

    # Concatenate the sampler type and dimensions to the script name
    script_name = f"{base_script_name}_{args.sampler}_dim{args.dimensions}_seed{args.seed}"

    # Set up logging and retrieve the log filename
    log_filename = set_logger(script_name, LOG_DIR)

    # Use log_filename as the storage name in DB_DIR
    storage_filename = os.path.splitext(log_filename)[0] + ".db"
    storage_path = os.path.join(DB_DIR, storage_filename)
    storage_url = f"sqlite:///{storage_path}"

    # Get the appropriate sampler based on command-line argument
    sampler = get_sampler(args.sampler, args.seed)

    # Define the experimental settings using argparse inputs
    settings = {
        "name": script_name,
        "seed": args.seed,
        "dimensions": args.dimensions,
        "iter_bo": args.iter_bo,  # Number of iterations for Bayesian optimization (ignored for Bruteforce)
        "storage": storage_url,  # Full path for the SQLite database in DB_DIR
        "sampler": sampler,  # Sampler passed to Optuna
        "acqf_settings": {
            "trade_off_param": args.acq_trade_off_param,  # Trade-off parameter for acquisition function
            "batch_size": args.acq_batch_size,  # Batch size for optimization
            "maximize": args.acq_maximize,  # Whether to maximize the acquisition function
        },
    }

    logging.info(f"Experiment settings: {settings}")

    # Run the Bayesian optimization experiment
    run_bo(settings)
