import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from scipy.stats import norm

from ..tf_grad import TensorFactorization


class TFSampler(BaseSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        method: Literal["cp", "tucker", "train", "ring"] = "cp",
        acquisition_function: Literal["ucb", "ei"] = "ucb",
        sampler_params: dict = {},
        tf_params: dict = {},
        acqf_params: dict = {},
        torch_device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tensor_constraint = None,
    ):
        # Random seed
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Device and dtype
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch_device
        if torch_dtype is None:
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        # Essential parameters
        self.method = method
        self.acquisition_function = acquisition_function
        self.independent_sampler = optuna.samplers.RandomSampler(seed=seed)

        # Sampler parameters
        self.n_startup_trials = sampler_params.get("n_startup_trials", 1)
        self.decomp_iter_num = sampler_params.get("decomp_iter_num", 10)
        self.decomp_parallel = sampler_params.get("decomp_parallel", False)
        self.mask_ratio = sampler_params.get("mask_ratio", 0.9)
        self.include_observed_points = sampler_params.get("include_observed_points", False)
        self.unique_sampling = sampler_params.get("unique_sampling", False)

        # Acquisition function parameters
        # self.acquisition_function = acqf_params.get("acquisition_function", "ucb")
        self.trade_off_param = acqf_params.get("trade_off_param", 1.0)
        self.batch_size = acqf_params.get("batch_size", 1) # Fixed to 1

        # TF optim parameters
        # self.method = tf_params.get("method", "cp")
        self.rank = tf_params.get("rank", 3)
        self.lr = tf_params.get("lr", 0.01)
        self.max_iter = tf_params.get("max_iter", None)
        self.tol = tf_params.get("tol", 1e-6)
        self.reg_lambda = tf_params.get("reg_lambda", 1e-3)
        self.constraint_lambda = tf_params.get("constraint_lambda", 1.0)
        # self.fill_constraint_method = tf_params.get("fill_constraint_method", "zero") # zero, normal or minmax

        # Internal storage
        self._param_names = None
        self._category_maps = None
        self._shape = None
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
        
        if len(trials) < self.n_startup_trials:
            return {}

        if self._param_names is None:
            self._initialize_internal_structure(search_space, study)

        # Build tensor from past trials
        self._update_tensor(study)

        # Perform CP decomposition and suggest next parameter set
        mean_tensor, std_tensor = self._fit(
            self._tensor_eval,
            self._tensor_eval_bool,
            parallel=self.decomp_parallel,
        )

        # Suggest next indices based on the selected acquisition function
        if self.acquisition_function == "ucb":
            next_indices = self._suggest_ucb_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                trade_off_param=self.trade_off_param,
                batch_size=self.batch_size,
                maximize=self._maximize,
            )
        elif self.acquisition_function == "ei":
            next_indices = self._suggest_ei_candidates(
                mean_tensor=mean_tensor,
                std_tensor=std_tensor,
                batch_size=self.batch_size,
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
        return self.independent_sampler.sample_independent(study, trial, param_name, param_distribution)

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
        self._maximize = StudyDirection.MAXIMIZE

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

    # def _fit(
    #     self,
    #     tensor_eval: np.ndarray,
    #     tensor_eval_bool: np.ndarray,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     eval_mean, eval_std = self._calculate_eval_stats(
    #         tensor_eval
    #     )

    #     tensors_list = [
    #         self._decompose_with_optional_mask(
    #             tensor_eval,
    #             tensor_eval_bool,
    #             eval_mean,
    #             eval_std,
    #             self._maximize
    #         )
    #         for _ in range(self.decomp_iter_num)
    #     ]

    #     return self._calculate_mean_std_tensors(
    #         tensors_list, tensor_eval, tensor_eval_bool, self._maximize
    #     )

    def _fit(
        self,
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        parallel: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        eval_mean, eval_std = self._calculate_eval_stats(tensor_eval)

        if parallel:
            # Run decompositions in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._decompose_with_optional_mask,
                        tensor_eval,
                        tensor_eval_bool,
                        eval_mean,
                        eval_std,
                        self._maximize
                    ) 
                    for _ in range(self.decomp_iter_num)
                ]
                tensors_list = [f.result() for f in futures]
        else:
            # Sequential execution
            tensors_list = [
                self._decompose_with_optional_mask(
                    tensor_eval,
                    tensor_eval_bool,
                    eval_mean,
                    eval_std,
                    self._maximize
                )
                for _ in range(self.decomp_iter_num)
            ]

        return self._calculate_mean_std_tensors(
            tensors_list,
            tensor_eval,
            tensor_eval_bool,
            self._maximize
        )

    def _calculate_eval_stats(
        self, tensor_eval: np.ndarray
    ) -> tuple[float, float]:
        eval_copy = np.copy(tensor_eval)
        # Filter with constraint
        if self._tensor_constraint is not None:
            eval_copy[self._tensor_constraint == 0] = np.nan

        finite_values = eval_copy[np.isfinite(eval_copy)]

        return (
            np.nanmean(finite_values),
            np.nanstd(finite_values),
        )

    def _select_mask_indices(
        self, tensor_shape: tuple, tensor_eval_bool: np.ndarray
    ) -> np.ndarray:
        if self._tensor_constraint is not None:
            # Get candidate indices where self._tensor_constraint is 1
            constrained_indices = np.argwhere(self._tensor_constraint == 1)

            # Filter candidate indices based on whether to include observed points
            if self.include_observed_points:
                cand_indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices = np.argwhere(tensor_eval_bool == False)

            # Intersect the constrained indices with the candidate indices
            constrained_indices_set = set(map(tuple, constrained_indices))
            cand_indices = np.array([idx for idx in cand_indices if tuple(idx) in constrained_indices_set])
        else:
            # If there is no tensor constraint, use all candidate indices
            if self.include_observed_points:
                cand_indices = np.indices(tensor_shape).reshape(len(tensor_shape), -1).T
            else:
                cand_indices = np.argwhere(tensor_eval_bool == False)

        # Determine the mask size
        mask_size = max(1, int(len(cand_indices) * self.mask_ratio))

        # Select mask indices
        selected_indices = self.rng.choice(len(cand_indices), mask_size, replace=False)
        return cand_indices[selected_indices]

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
        maximize: bool
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

        # Assign observed values
        if self._tensor_constraint is not None:
            condition = np.logical_and(tensor_eval_bool, self._tensor_constraint)
        else:
            condition = tensor_eval_bool
        init_tensor_eval[condition] = standardized_tensor_eval[condition]

        # # Incorporate constraint based on method
        # if self._tensor_constraint is not None:
        #     if self.fill_constraint_method == "zero":
        #         # Fill constrained values with zeros
        #         init_tensor_eval[self._tensor_constraint == 0] = 0.0
        #     if self.fill_constraint_method == "normal":
        #         # Fill constrained values with random normal samples
        #         init_tensor_eval[self._tensor_constraint == 0] = self.rng.normal(0, 1, np.sum(self._tensor_constraint == 0))
        #     elif self.fill_constraint_method == "minmax":
        #         # Fill constrained values based on min/max
        #         if maximize:
        #             init_tensor_eval[self._tensor_constraint == 0] = np.nanmin(init_tensor_eval) - 1.0
        #         else:
        #             init_tensor_eval[self._tensor_constraint == 0] = np.nanmax(init_tensor_eval) + 1.0

        if maximize:
            init_tensor_eval[self._tensor_constraint == 0] = np.nanmin(init_tensor_eval) - 1.0
        else:
            init_tensor_eval[self._tensor_constraint == 0] = np.nanmax(init_tensor_eval) + 1.0

        if self._tensor_constraint is not None:
            constraint = torch.tensor(self._tensor_constraint, dtype=self.torch_dtype)
        else:
            constraint = None
            
        tf = TensorFactorization(
            tensor=torch.tensor(init_tensor_eval, dtype=self.torch_dtype),
            rank=self.rank,
            method=self.method,
            mask=torch.tensor(mask_tensor, dtype=self.torch_dtype),
            constraint=constraint,
            device=self.torch_device,
        )
        tf.optimize(
            lr=self.lr,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_lambda=self.reg_lambda,
            constraint_lambda=self.constraint_lambda,
            verbose=False,
        )
        reconstructed_tensor = tf.reconstruct()

        return reconstructed_tensor.detach().cpu().numpy()

    def _calculate_mean_std_tensors(
        self,
        tensors_list: list[np.ndarray],
        tensor_eval: np.ndarray,
        tensor_eval_bool: np.ndarray,
        maximize: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        tensors_stack = np.stack(tensors_list)
        mean_tensor = np.mean(tensors_stack, axis=0)
        std_tensor = np.std(tensors_stack, axis=0)

        mean_tensor[tensor_eval_bool] = tensor_eval[tensor_eval_bool]
        std_tensor[tensor_eval_bool] = 0

        if self._tensor_constraint is not None:
            # if maximize:
            #     mean_tensor[self._tensor_constraint == 0] = np.min(mean_tensor) - 1.0
            # else:
            #     mean_tensor[self._tensor_constraint == 0] = np.max(mean_tensor) + 1.0
                
            std_tensor[self._tensor_constraint == 0] = 0

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