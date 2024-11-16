import logging
import random
from typing import Literal

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac
from tensorly.decomposition import non_negative_parafac


class ParafacSampler(BaseSampler):
    def __init__(
        self,
        cp_rank: int = 3,
        als_iter_num: int = 10,
        mask_ratio: float = 0.2,
        trade_off_param: float = 1.0,
        distribution_type: Literal["normal", "uniform"] = "normal",
        seed: int = None,
        unique_sampling: bool = False,
        decomp_iter_num: int = 5,
        include_observed_points: bool = True,
    ):
        # Initialization
        self.cp_rank = cp_rank
        self.als_iter_num = als_iter_num
        self.mask_ratio = mask_ratio
        self.trade_off_param = trade_off_param
        self.distribution_type = distribution_type
        self.rng = np.random.RandomState(seed)
        self.unique_sampling = unique_sampling
        self.decomp_iter_num = decomp_iter_num
        self.include_observed_points = include_observed_points

        # Internal storage
        self._param_names = None
        self._category_maps = None
        self._shape = None
        self._evaluated_indices = []
        self._tensor_eval = None
        self._tensor_eval_bool = None
        self._maximize = None

        ##################################################
        ##################################################
        ##################################################
        ##################################################
        # Debuging
        self.mean_tensor = None
        self.std_tensor = None
        # self.save_dir = "tensor_dir"
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

        if self._param_names is None:
            self._initialize_internal_structure(search_space, study)

        # Build tensor from past trials
        self._update_tensor(study)

        # Perform CP decomposition and suggest next parameter set
        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool,
            self._evaluated_indices,
            self.distribution_type,
        )

        # Suggest next indices based on UCB
        next_indices = self._suggest_ucb_candidates(
            mean_tensor=mean_tensor,
            std_tensor=std_tensor,
            trade_off_param=self.trade_off_param,
            batch_size=1,
            maximize=self._maximize,
        )

        # ここでランダムに選んでも意味がない
        # # Randomly select next indices
        # next_index = random.sample(next_indices, 1)[0]
        next_index = next_indices[0]

        # Convert indices back to parameter values
        params = {}
        for i, param_name in enumerate(self._param_names):
            category_index = next_index[i]
            category = self._category_maps[param_name][category_index]
            params[param_name] = category

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        logging.info(f"Using sample_independent for sampling.")
        return optuna.samplers.RandomSampler().sample_independent(
            study, trial, param_name, param_distribution
        )

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

        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        trial_num = trial.number - 1
        if trial_num >= 1:
            self._save_tensor(self._tensor_eval, "tensor_eval", trial_num)
            self._save_tensor(self._tensor_eval_bool, "tensor_eval_bool", trial_num)
            self._save_tensor(self.mean_tensor, "mean_tensor", trial_num)
            self._save_tensor(self.std_tensor, "std_tensor", trial_num)
    
    def _save_tensor(self, tensor: np.ndarray, name: str, trial_index: int):
        """
        Save a tensor to the specified directory with a given name and trial index.
        """
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
        all_evaluated_indices: list[tuple[int, ...]],
        distribution_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        eval_min, eval_max, eval_mean, eval_std = self._calculate_eval_stats(
            tensor_eval
        )

        tensors_list = [
            self._decompose_with_optional_mask(
                tensor_eval,
                tensor_eval_bool,
                eval_min,
                eval_max,
                eval_mean,
                eval_std,
                distribution_type,
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
            np.nanmin(finite_values),
            np.nanmax(finite_values),
            np.nanmean(finite_values),
            np.nanstd(finite_values),
        )

    def _generate_random_array(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        distribution_type: str = "uniform",
        mean: float = 0,
        std_dev: float = 1,
    ) -> np.ndarray:
        if low == high:
            low = low - 1e-1
            high = high + 1e-1
            std_dev = std_dev + 1e-1

        if distribution_type == "uniform":
            return self.rng.uniform(low, high, shape)
        elif distribution_type == "normal":
            normal_random = self.rng.normal(mean, std_dev, shape)
            return np.clip(normal_random, low, high)
        else:
            raise ValueError("distribution_type must be either 'uniform' or 'normal'.")

    def _select_mask_indices(
        self, tensor_shape: tuple, tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        cand_indices = (
            np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            if self.include_observed_points
            else np.argwhere(tensor_eval_bool == False)
        )
        mask_size = max(1, int(len(cand_indices) * self.mask_ratio))
        return random.sample(list(cand_indices), mask_size)

    def _create_mask_tensor(
        self, tensor_shape: tuple, mask_indices: np.ndarray
    ) -> np.ndarray:
        mask_tensor = np.ones(tensor_shape, dtype=bool)
        for mask_index in mask_indices:
            mask_tensor[tuple(mask_index)] = False
        return mask_tensor

    # def _decompose_with_optional_mask(
    #     self,
    #     tensor_eval: np.ndarray,
    #     tensor_eval_bool: np.ndarray,
    #     eval_min: float,
    #     eval_max: float,
    #     eval_mean: float,
    #     eval_std: float,
    #     distribution_type: str,
    # ) -> np.ndarray:
    #     mask_tensor = None
    #     if self.mask_ratio != 0:
    #         mask_indices = self._select_mask_indices(
    #             tensor_eval.shape, tensor_eval_bool
    #         )
    #         mask_tensor = self._create_mask_tensor(tensor_eval.shape, mask_indices)

    #     init_tensor_eval = self._generate_random_array(
    #         low=eval_min,
    #         high=eval_max,
    #         shape=tensor_eval.shape,
    #         distribution_type=distribution_type,
    #         mean=eval_mean,
    #         std_dev=eval_std,
    #     )
    #     init_tensor_eval[tensor_eval_bool] = tensor_eval[tensor_eval_bool]

    #     cp_tensor = parafac(
    #         init_tensor_eval,
    #         rank=self.cp_rank,
    #         mask=mask_tensor,
    #         n_iter_max=self.als_iter_num,
    #         init="random",
    #         random_state=self.rng,
    #     )
    #     return cp_to_tensor(cp_tensor)

    def _decompose_with_optional_mask(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        eval_min: float,
        eval_max: float,
        eval_mean: float,
        eval_std: float,
        distribution_type: str,
    ) -> np.ndarray:
        # 標準化された tensor_eval を作成
        standardized_tensor_eval = (tensor_eval - eval_mean) / (eval_std + 1e-8)  # eval_stdが0の場合のために小さな値を足す
        standardized_tensor_eval[~tensor_eval_bool] = np.nan  # 未観測点を NaN に設定

        # マスクが必要な場合に設定
        mask_tensor = None
        if self.mask_ratio != 0:
            mask_indices = self._select_mask_indices(
                tensor_eval.shape, tensor_eval_bool
            )
            mask_tensor = self._create_mask_tensor(tensor_eval.shape, mask_indices)

        # 標準正規分布に従って未観測点をランダムに生成
        init_tensor_eval = self._generate_random_array(
            low=-1,  # 標準化後の値の範囲として適当な値を設定
            high=1,
            shape=tensor_eval.shape,
            distribution_type="normal",  # 標準正規分布
            mean=0,
            std_dev=1,
        )
        
        # 観測点に標準化された値を代入
        init_tensor_eval[tensor_eval_bool] = standardized_tensor_eval[tensor_eval_bool]

        # CP分解を実行
        cp_tensor = parafac(
            init_tensor_eval,
            rank=self.cp_rank,
            mask=mask_tensor,
            n_iter_max=self.als_iter_num,
            init="random",
            random_state=self.rng,
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


        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################
        ##################################################

        # Debuging
        self.mean_tensor = mean_tensor
        self.std_tensor = std_tensor

        return mean_tensor, std_tensor
    
    # def _calculate_mean_std_tensors(
    #     self,
    #     tensors_list: list[np.ndarray],
    #     tensor_eval: np.ndarray,
    #     tensor_eval_bool: np.ndarray,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     # Stack the list of tensors into a 3D array
    #     tensors_stack = np.stack(tensors_list)




    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     ##################################################
    #     # 多分平均と分散の埋め方が間違っている（というかそうであってほしい）





    #     # Calculate mean and standard deviation across tensors for standardization
    #     stack_mean = np.mean(tensors_stack, axis=0, keepdims=True)
    #     stack_std = np.std(tensors_stack, axis=0, keepdims=True)
    #     standardized_stack = (tensors_stack - stack_mean) / (stack_std + 1e-8)  # Avoid division by zero

    #     # Calculate mean and standard deviation from the standardized tensors
    #     mean_tensor = np.mean(standardized_stack, axis=0)
    #     std_tensor = np.std(standardized_stack, axis=0)

    #     # Replace elements in evaluated positions
    #     mean_tensor[tensor_eval_bool] = tensor_eval[tensor_eval_bool]
    #     std_tensor[tensor_eval_bool] = 0

    #     return mean_tensor, std_tensor

    def _suggest_ucb_candidates(
        self,
        mean_tensor: np.ndarray,
        std_tensor: np.ndarray,
        trade_off_param: float,
        batch_size: int,
        maximize: bool,
    ) -> list[tuple[int, ...]]:
        
        # Calculate overall mean and std stats for logging
        mean_stats = {
            "Max": np.max(mean_tensor),
            "Min": np.min(mean_tensor),
            "Mean": np.mean(mean_tensor),
            "Std": np.std(mean_tensor),
        }
        std_stats = {
            "Max": np.max(std_tensor),
            "Min": np.min(std_tensor),
            "Mean": np.mean(std_tensor),
            "Std": np.std(std_tensor),
        }

        logging.info(f"Candidate Mean Stats: {mean_stats}")
        logging.info(f"Candidate Std Stats: {std_stats}")
        
        # Define UCB calculation
        def _ucb(mean_tensor, std_tensor, trade_off_param, maximize=True) -> np.ndarray:
            mean_tensor = mean_tensor if maximize else -mean_tensor
            ucb_values = mean_tensor + trade_off_param * std_tensor
            return ucb_values

        ucb_values = _ucb(mean_tensor, std_tensor, trade_off_param, maximize)

        if self.unique_sampling:
            ucb_values[self._tensor_eval_bool == True] = -np.inf

        # Get indices of top and bottom UCB values
        flat_indices = np.argsort(ucb_values.flatten())[::-1]
        top_indices = np.unravel_index(flat_indices[:10], ucb_values.shape)
        top_indices = list(zip(*top_indices))

        bottom_indices = np.unravel_index(flat_indices[-10:], ucb_values.shape)
        bottom_indices = list(zip(*bottom_indices))

        # Calculate Mean and Std stats for top 10 UCB values
        top_means = [mean_tensor[idx] for idx in top_indices]
        top_stds = [std_tensor[idx] for idx in top_indices]
        top_mean_stats = {
            "Max": np.max(top_means),
            "Min": np.min(top_means),
            "Mean": np.mean(top_means),
            "Std": np.std(top_means),
        }
        top_std_stats = {
            "Max": np.max(top_stds),
            "Min": np.min(top_stds),
            "Mean": np.mean(top_stds),
            "Std": np.std(top_stds),
        }
        logging.info(f"Top 10 Mean Stats: {top_mean_stats}")
        logging.info(f"Top 10 Std Stats: {top_std_stats}")

        # Calculate Mean and Std stats for bottom 10 UCB values
        bottom_means = [mean_tensor[idx] for idx in bottom_indices]
        bottom_stds = [std_tensor[idx] for idx in bottom_indices]
        bottom_mean_stats = {
            "Max": np.max(bottom_means),
            "Min": np.min(bottom_means),
            "Mean": np.mean(bottom_means),
            "Std": np.std(bottom_means),
        }
        bottom_std_stats = {
            "Max": np.max(bottom_stds),
            "Min": np.min(bottom_stds),
            "Mean": np.mean(bottom_stds),
            "Std": np.std(bottom_stds),
        }
        logging.info(f"Bottom 10 Mean Stats: {bottom_mean_stats}")
        logging.info(f"Bottom 10 Std Stats: {bottom_std_stats}")

        # Select top candidates based on batch size
        top_indices = np.unravel_index(flat_indices[:batch_size], ucb_values.shape)
        top_indices = list(zip(*top_indices))

        return top_indices
