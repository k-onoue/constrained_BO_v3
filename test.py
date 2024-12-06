import logging
import random
from typing import Literal, Optional

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
from scipy.stats import norm
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac


class ParafacSampler(BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        cp_rank: int = 3,
        als_iter_num: int = 10,
        mask_ratio: float = 0.2,
        trade_off_param: float = 1.0,
        unique_sampling: bool = False,
        decomp_iter_num: int = 5,
        include_observed_points: bool = False,
        acquisition_function: Literal["ucb", "ei"] = "ucb",  # 追加: 獲得関数の選択
        n_startup_trials: int = 1,
        independent_sampler: Literal["random"] = "random",  # Remove qmc option
        tensor_constraint = None,
    ):
        # Random seed
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)  # Ensure the global random seed is set

        # Initialization
        self.cp_rank = cp_rank
        self.als_iter_num = als_iter_num
        self.mask_ratio = mask_ratio
        self.trade_off_param = trade_off_param
        self.unique_sampling = unique_sampling
        self.decomp_iter_num = decomp_iter_num
        self.include_observed_points = include_observed_points
        self.acquisition_function = acquisition_function  # 追加: 獲得関数の設定
        self.n_startup_trials = n_startup_trials
        self.independent_sampler = independent_sampler  # Store the independent sampler choice

        # Internal storage
        self._param_names = None
        self._category_maps = None
        self._shape = None
        self._evaluated_indices = []
        self._tensor_eval = None
        self._tensor_eval_bool = None
        self._tensor_constraint = tensor_constraint
        self._maximize = None

        # Debugging
        self.mean_tensor = None
        self.std_tensor = None
        self.save_dir = None

    def infer_relative_search_space(self, study, trial):
        search_space = optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )
        relevant_search_space = {}
        for name, distribution in search_space.items():
            if isinstance(
                distribution,
                (
                    optuna.distributions.IntDistribution,
                    optuna.distributions.CategoricalDistribution,
                ),
            ):
                relevant_search_space[name] = distribution
        return relevant_search_space

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        
        if len(trials) < self.n_startup_trials:  # Corrected attribute name
            return {}

        if self._param_names is None:
            self._initialize_internal_structure(search_space, study)

        # Build tensor from past trials
        self._update_tensor(study)

        # Perform CP decomposition and suggest next parameter set
        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool,
        )

        # Suggest next indices based on the selected acquisition function
        if self.acquisition_function == "ucb":
            next_indices = self._suggest_ucb_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                trade_off_param=self.trade_off_param,
                batch_size=1,
                maximize=self._maximize,
            )
        elif self.acquisition_function == "ei":
            next_indices = self._suggest_ei_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                batch_size=1,
                maximize=self._maximize,
            )
        else:
            raise ValueError("acquisition_function must be either 'ucb' or 'ei'.")

        next_index = next_indices[0]

        # Convert indices back to parameter values
        params = {}
        for i, param_name in enumerate(self._param_names):
            category_index = next_index[i]
            category = self._category_maps[param_name][category_index]
            params[param_name] = category

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        logging.info(f"Using sample_independent for sampling with {self.independent_sampler} sampler.")
        if self.independent_sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)  # Pass the seed to the sampler
        else:
            raise ValueError("independent_sampler must be 'random'.")
        return sampler.sample_independent(study, trial, param_name, param_distribution)

    def _initialize_internal_structure(self, search_space, study):
        self._param_names = sorted(search_space.keys())
        self._category_maps = {}
        self._shape = []
        for param_name in self._param_names:
            distribution = search_space[param_name]
            if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                categories = distribution.choices
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                categories = list(
                    range(distribution.low, distribution.high + 1, distribution.step)
                )
            else:
                continue
            self._category_maps[param_name] = categories
            self._shape.append(len(categories))
        self._shape = tuple(self._shape)
        self._tensor_eval = np.full(self._shape, np.nan)
        self._tensor_eval_bool = np.zeros(self._shape, dtype=bool)
        self._evaluated_indices = []
        self._maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

    def _update_tensor(self, study):
        trials = study.get_trials(deepcopy=False)
        for trial in trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            index = []
            for param_name in self._param_names:
                if param_name not in trial.params:
                    break
                category = trial.params[param_name]
                try:
                    category_index = self._category_maps[param_name].index(category)
                except ValueError:
                    break
                index.append(category_index)
            else:
                index = tuple(index)
                if index not in self._evaluated_indices:
                    value = trial.value
                    self._tensor_eval[index] = value
                    self._tensor_eval_bool[index] = True
                    self._evaluated_indices.append(index)

        # Debugging (optional saving)
        trial_num = trial.number - 1
        if trial_num >= 1:
            self._save_tensor(self._tensor_eval, "tensor_eval", trial_num)
            self._save_tensor(self._tensor_eval_bool, "tensor_eval_bool", trial_num)
            self._save_tensor(self.mean_tensor, "mean_tensor", trial_num)
            self._save_tensor(self.std_tensor, "std_tensor", trial_num)

    def _save_tensor(self, tensor: np.ndarray, name: str, trial_index: int):
        import os
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            filepath = os.path.join(self.save_dir, f"{name}_trial{trial_index}.npy")
            np.save(filepath, tensor)
            print(f"Saved {name} for trial {trial_index} at {filepath}")

    def _fit(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eval_mean, eval_std = self._calculate_eval_stats(
            tensor_eval
        )

        tensors_list = [
            self._decompose_with_optional_mask(
                tensor_eval,
                tensor_eval_bool,
                eval_mean,
                eval_std,
            )
            for _ in range(self.decomp_iter_num)
        ]

        return self._calculate_mean_std_tensors(
            tensors_list, tensor_eval, tensor_eval_bool
        )

    def _calculate_eval_stats(
        self, tensor_eval: np.ndarray
    ) -> tuple[float, float, float, float]:
        finite_values = tensor_eval[np.isfinite(tensor_eval)]
        return (
            np.nanmean(finite_values),
            np.nanstd(finite_values),
        )

    def _select_mask_indices(
        self, tensor_shape: tuple, tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        cand_indices = (
            np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            if self.include_observed_points
            else np.argwhere(tensor_eval_bool == False)
        )
        mask_size = max(1, int(len(cand_indices) * self.mask_ratio))
        # return self.rng.choice(cand_indices, mask_size, replace=False)  # Use self.rng.choice instead of random.sample
        return random.sample(list(cand_indices), mask_size)

    def _create_mask_tensor(
        self, tensor_shape: tuple, mask_indices: np.ndarray
    ) -> np.ndarray:
        mask_tensor = np.ones(tensor_shape, dtype=bool)
        for mask_index in mask_indices:
            mask_tensor[tuple(mask_index)] = False
        return mask_tensor

    def _decompose_with_optional_mask(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        eval_mean: float,
        eval_std: float,
    ) -> np.ndarray:
        # Standardize tensor_eval
        standardized_tensor_eval = (tensor_eval - eval_mean) / (eval_std + 1e-8)
        standardized_tensor_eval[~tensor_eval_bool] = np.nan  # Set unobserved points to NaN

        # Create mask if needed
        mask_tensor = None
        if self.mask_ratio != 0:
            mask_indices = self._select_mask_indices(
                tensor_eval.shape, tensor_eval_bool
            )
            mask_tensor = self._create_mask_tensor(tensor_eval.shape, mask_indices)

        init_tensor_eval = self.rng.normal(0, 1, tensor_eval.shape)

        # Apply constraints
        if self._tensor_constraint:
            init_tensor_eval = np.where(
                self._tensor_constraint == 0, 
                -100 if self._maximize else 100, 
                init_tensor_eval
            )

        # Assign observed values
        init_tensor_eval[tensor_eval_bool] = standardized_tensor_eval[tensor_eval_bool]

        # Perform CP decomposition
        cp_tensor = parafac(
            init_tensor_eval,
            rank=self.cp_rank,
            mask=mask_tensor,
            n_iter_max=self.als_iter_num,
            init="random",
            random_state=self.rng,  # Ensure the random state is passed
        )
        return cp_to_tensor(cp_tensor)

    def _calculate_mean_std_tensors(
        self,
        tensors_list: list[np.ndarray],
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0)
        mean_tensor[tensor_eval_bool] = tensor_eval[tensor_eval_bool]
        std_tensor[tensor_eval_bool] = 0

        # Debugging
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

        return mean_tensor, std_tensor

    def _suggest_ucb_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        trade_off_param: float,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        # # Calculate overall mean and std stats for logging
        # mean_stats = {
        #     "Max": np.max(mean_tensor),
        #     "Min": np.min(mean_tensor),
        #     "Mean": np.mean(mean_tensor),
        #     "Std": np.std(mean_tensor),
        # }
        # std_stats = {
        #     "Max": np.max(std_tensor),
        #     "Min": np.min(std_tensor),
        #     "Mean": np.mean(std_tensor),
        #     "Std": np.std(std_tensor),
        # }

        # logging.info(f"Candidate Mean Stats: {mean_stats}")
        # logging.info(f"Candidate Std Stats: {std_stats}")

        # Define UCB calculation
        def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
            mean_tensor = mean_tensor if maximize else -mean_tensor
            ucb_values = mean_tensor + trade_off_param * std_tensor
            return ucb_values

        ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

        if self.unique_sampling:
            ucb_values[self._tensor_eval_bool == True] = -np.inf

        # Get indices of top UCB values
        flat_indices = np.argsort(ucb_values.flatten())[::-1]
        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        top_indices = list(zip(*top_indices))

        return top_indices

    def _suggest_ei_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        # # Calculate overall mean and std stats for logging
        # mean_stats = {
        #     "Max": np.max(mean_tensor),
        #     "Min": np.min(mean_tensor),
        #     "Mean": np.mean(mean_tensor),
        #     "Std": np.std(mean_tensor),
        # }
        # std_stats = {
        #     "Max": np.max(std_tensor),
        #     "Min": np.min(std_tensor),
        #     "Mean": np.mean(std_tensor),
        #     "Std": np.std(std_tensor),
        # }

        # Define EI calculation
        def _ei(mean_tensor, std_tensor, f_best, maximize=True) -> np.ndarray:
            std_tensor = np.clip(std_tensor, 1e-9, None)
            if maximize:
                z = (mean_tensor - f_best) / std_tensor
            else:
                z = (f_best - mean_tensor) / std_tensor
            ei_values = std_tensor * (z * norm.cdf(z) + norm.pdf(z))
            return ei_values

        if maximize:
            f_best = np.nanmax(self._tensor_eval)
        else:
            f_best = np.nanmin(self._tensor_eval)

        ei_values = _ei(mean_tensor, std_tensor, f_best=f_best, maximize=maximize)

        if self.unique_sampling:
            ei_values[self._tensor_eval_bool == True] = -np.inf

        # Get indices of top EI values
        flat_indices = np.argsort(ei_values.flatten())[::-1]
        top_indices = np.unravel_index(flat_indices[:batch_size], ei_values.shape)
        top_indices = list(zip(*top_indices))

        return top_indices
    


import random
import numpy as np
import optuna
from functools import partial
from src.objectives.warcraft import WarcraftObjective


def build_constraint_warcraft(map_shape: tuple[int, int]) -> np.ndarray:
    # Directions dictionary
    directions_dict = {
        "oo": np.array([0, 0]),
        "ab": np.array([1, 1]),
        "ac": np.array([0, 2]),
        "ad": np.array([1, 1]),
        "bc": np.array([1, 1]),
        "bd": np.array([2, 0]),
        "cd": np.array([1, 1]),
    }

    directions_list = list(directions_dict.keys())

    # Map parameters
    map_length = map_shape[0] * map_shape[1]
    ideal_gain = (map_shape[0] + map_shape[1] - 1) * 2

    # Initialize constraints as NumPy arrays
    tensor_constraint_1 = np.zeros((len(directions_list),) * map_length)
    tensor_constraint_2 = np.zeros((len(directions_list),) * map_length)
    tensor_constraint_3 = np.zeros((len(directions_list),) * map_length)

    # Constraint 1: (0, 0) != "oo", "ab"
    for direction in directions_list:
        if direction not in ["oo", "ab"]:
            tensor_constraint_1[..., directions_list.index(direction)] = 1

    # Constraint 2: (map_shape[0] - 1, map_shape[1] - 1) != "oo", "cd"
    for direction in directions_list:
        if direction not in ["oo", "cd"]:
            tensor_constraint_2[directions_list.index(direction), ...] = 1

    # Constraint 3: len[path] == map_shape[0] * map_shape[1]
    for index, _ in np.ndenumerate(tensor_constraint_3):
        gain = np.sum([directions_dict[directions_list[idx]].sum() for idx in index])
        if gain == ideal_gain:
            tensor_constraint_3[index] = 1

    # Combine constraints with logical AND
    tensor_constraint = np.logical_and(
        tensor_constraint_1,
        np.logical_and(tensor_constraint_2, tensor_constraint_3)
    )

    return tensor_constraint


def objective(trial, map_shape=None, objective_function=None):
    directions = ["oo", "ab", "ac", "ad", "bc", "bd", "cd"]
    x = np.empty(map_shape, dtype=object)
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            x[i, j] = trial.suggest_categorical(f"x_{i}_{j}", directions)
    return objective_function(x)


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


def run_bo():
    settings = {
        "name": "test_study",
        "seed": 0,
        "map_option": 1,
        "iter_bo": 10,
        "unique_sampling": False,
        "decomp_num": 5,
        "cp_settings": {
            "rank": 2,
            "als_iterations": 100,
            "mask_ratio": 0.1,
            "include_observed_points": False,
        },
        "acqf_settings": {
            "acquisition_function": "ucb",
            "trade_off_param": 3.0,
            "batch_size": 10,
            "maximize": True,
        },
        "n_startup_trials": 10,
    }

    random.seed(settings['seed'])

    map_targeted = get_map(settings["map_option"])
    map_shape = map_targeted.shape
    objective_function = WarcraftObjective(map_targeted)
    objective_with_args = partial(objective, map_shape=map_shape, objective_function=objective_function)

    sampler = ParafacSampler(
        cp_rank=settings["cp_settings"]["rank"],
        als_iter_num=settings["cp_settings"]["als_iterations"],
        mask_ratio=settings["cp_settings"]["mask_ratio"],
        acquisition_function=settings["acqf_settings"]["acquisition_function"],
        trade_off_param=settings["acqf_settings"]["trade_off_param"],
        seed=settings["seed"],
        unique_sampling=settings["unique_sampling"],
        decomp_iter_num=settings["decomp_num"],
        include_observed_points=settings["cp_settings"].get("include_observed_points", False),
        tensor_constraint=build_constraint_warcraft(map_shape),
    )

    direction = "maximize" if settings["acqf_settings"]["maximize"] else "minimize"

    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
    )

    study.optimize(objective_with_args, n_trials=settings["iter_bo"])

    best_x = np.empty(map_shape, dtype=object)
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            best_x[i, j] = study.best_params[f"x_{i}_{j}"]

    optuna.visualization.plot_optimization_history(study)



if __name__ == "__main__":
    run_bo()