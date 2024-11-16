import subprocess
from datetime import datetime
from functools import partial

def run(function, timestamp, sampler="hgp", dimension=2, iter_bo=10, seed=0, map_option=1):
    cmd = [
        "python3", "experiments/benchmark.py",
        "--timestamp", timestamp,
        "--function", function,
        "--sampler", sampler,
        "--dimension", str(dimension),
        "--iter_bo", str(iter_bo),
        "--seed", str(seed),
        "--map_option", str(map_option)
    ]
    subprocess(cmd)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_with_timestamp = partial(run, timestamp=timestamp)

    objective_function_dict = {
        "sphere": {"dimensions": [2, 3, 4, 5, 6, 7]},
        "ackley": {"dimensions": [2, 3, 4, 5, 6, 7]},
        "warcraft": {"map_options": [1, 2, 3]}
    }
    sampler_list = ["tpe", "random", "gp", "hgp"]
    seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    iter_bo = 20

    for function, params in objective_function_dict.items():
        if function in ["sphere", "ackley"]:
            for dimension in params["dimensions"]:
                for sampler in sampler_list:
                    for seed in seed_list:
                        run_with_timestamp(
                            function=function,
                            dimension=dimension,
                            map_option=1,  # Default map_option for non-warcraft functions
                            sampler=sampler,
                            seed=seed,
                            iter_bo=iter_bo
                        )
        elif function == "warcraft":
            for map_option in params["map_options"]:
                for sampler in sampler_list:
                    for seed in seed_list:
                        run_with_timestamp(
                            function=function,
                            dimension=2,  # Default dimension for warcraft
                            map_option=map_option,
                            sampler=sampler,
                            seed=seed,
                            iter_bo=iter_bo
                        )
