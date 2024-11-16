from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any, Sequence, cast

import numpy as np
import optuna
from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.logging import get_logger


import torch
import scipy.optimize as so
import optuna._gp.acqf as acqf
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.search_space as gp_search_space
from optuna.study import Study


logger = get_logger(__name__)

class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: typing.Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        (deriv,) = ctx.saved_tensors
        return deriv * grad

def matern52_kernel_from_squared_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    return Matern52Kernel.apply(squared_distance)  # type: ignore

@dataclass(frozen=True)
class KernelParamsTensor:
    inverse_squared_lengthscales: torch.Tensor  # [len(params)]
    kernel_scale: torch.Tensor  # Scalar
    noise_var: torch.Tensor  # Scalar

def kernel(
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
    X1: torch.Tensor,  # [...batch_shape, n_A, len(params)]
    X2: torch.Tensor,  # [...batch_shape, n_B, len(params)]
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
    
    d2 = (d2 * kernel_params.inverse_squared_lengthscales).sum(dim=-1)
    return matern52_kernel_from_squared_distance(d2) * kernel_params.kernel_scale

def kernel_at_zero_distance(
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    return kernel_params.kernel_scale

def posterior(
    kernel_params: KernelParamsTensor,
    X: torch.Tensor,  # [len(trials), len(params)]
    is_categorical: torch.Tensor,  # bool[len(params)]
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    x: torch.Tensor,  # [(batch,) len(params)]
) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: [(batch,)], var: [(batch,)])
    cov_fx_fX = kernel(is_categorical, kernel_params, x[..., None, :], X)[..., 0, :]
    cov_fx_fx = kernel_at_zero_distance(kernel_params)
    mean = cov_fx_fX @ cov_Y_Y_inv_Y  # [batch]
    var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)  # [batch]
    return (mean, torch.clamp(var, min=0.0))

def marginal_log_likelihood(
    X: torch.Tensor,  # [len(trials), len(params)]
    Y: torch.Tensor,  # [len(trials)]
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParamsTensor,
) -> torch.Tensor:  # Scalar
    cov_fX_fX = kernel(is_categorical, kernel_params, X, X)
    cov_Y_Y_chol = torch.linalg.cholesky(
        cov_fX_fX + kernel_params.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
    )
    logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
    cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    return -0.5 * (
        logdet
        + X.shape[0] * math.log(2 * math.pi)
        + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
    )

def _fit_kernel_params(
    X: np.ndarray,  # [len(trials), len(params)]
    Y: np.ndarray,  # [len(trials)]
    is_categorical: np.ndarray,  # [len(params)]
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel_params: KernelParamsTensor,
    gtol: float,
) -> KernelParamsTensor:
    n_params = X.shape[1]
    initial_raw_params = np.concatenate(
        [
            np.log(initial_kernel_params.inverse_squared_lengthscales.detach().numpy()),
            [
                np.log(initial_kernel_params.kernel_scale.item()),
                np.log(initial_kernel_params.noise_var.item() - 0.99 * minimum_noise),
            ],
        ]
    )

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        params = KernelParamsTensor(
            inverse_squared_lengthscales=torch.exp(raw_params_tensor[:n_params]),
            kernel_scale=torch.exp(raw_params_tensor[n_params]),
            noise_var=(
                torch.tensor(minimum_noise, dtype=torch.float64)
                if deterministic_objective
                else torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise
            ),
        )
        loss = -marginal_log_likelihood(
            torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical), params
        ) - log_prior(params)
        loss.backward()  # type: ignore
        raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
        assert not deterministic_objective or raw_noise_var_grad == 0
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    res = so.minimize(
        loss_func,
        initial_raw_params,
        jac=True,
        method="l-bfgs-b",
        options={"gtol": gtol},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    raw_params_opt_tensor = torch.from_numpy(res.x)

    res = KernelParamsTensor(
        inverse_squared_lengthscales=torch.exp(raw_params_opt_tensor[:n_params]),
        kernel_scale=torch.exp(raw_params_opt_tensor[n_params]),
        noise_var=(
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        ),
    )
    return res

def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[KernelParamsTensor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    initial_kernel_params: KernelParamsTensor | None = None,
    gtol: float = 1e-2,
) -> KernelParamsTensor:
    default_initial_kernel_params = KernelParamsTensor(
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
    )
    if initial_kernel_params is None:
        initial_kernel_params = default_initial_kernel_params

    error = None
    for init_kernel_params in [initial_kernel_params, default_initial_kernel_params]:
        try:
            return _fit_kernel_params(
                X=X,
                Y=Y,
                is_categorical=is_categorical,
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                initial_kernel_params=init_kernel_params,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warn(
        f"The optimization of kernel_params failed: \n{error}\n"
        "The default initial kernel params will be used instead."
    )
    return default_initial_kernel_params

@experimental_class("3.6.0")
class CustomGPSampler(BaseSampler):
    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: "Callable[[KernelParamsTensor], torch.Tensor]" = (
            lambda _: 0
        )
        self._minimum_noise: float = 1e-10
        self._kernel_params_cache: "KernelParamsTensor | None" = None
        self._optimize_n_samples: int = 2048
        self._deterministic = deterministic_objective

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _optimize_acqf(
        self,
        acqf_params: "acqf.AcquisitionFunctionParams",
        best_params: np.ndarray,
    ) -> np.ndarray:
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf_params,
            warmstart_normalized_params_array=best_params[None, :],
            n_preliminary_samples=2048,
            n_local_search=10,
            tol=1e-4,
            rng=self._rng.rng,
        )
        return normalized_params

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        score_vals = np.array([_sign * cast(float, trial.value) for trial in trials])

        if np.any(~np.isfinite(score_vals)):
            warnings.warn(
                "GPSampler cannot handle infinite values. "
                "We clamp those values to worst/best finite value."
            )

            finite_score_vals = score_vals[np.isfinite(score_vals)]
            best_finite_score = np.max(finite_score_vals, initial=0.0)
            worst_finite_score = np.min(finite_score_vals, initial=0.0)

            score_vals = np.clip(score_vals, worst_finite_score, best_finite_score)

        standarized_score_vals = (score_vals - score_vals.mean()) / max(1e-10, score_vals.std())

        if self._kernel_params_cache is not None and len(
            self._kernel_params_cache.inverse_squared_lengthscales
        ) != len(internal_search_space.scale_types):
            self._kernel_params_cache = None

        kernel_params = fit_kernel_params(
            X=normalized_params,
            Y=standarized_score_vals,
            is_categorical=(
                internal_search_space.scale_types == gp_search_space.ScaleType.CATEGORICAL
            ),
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            initial_kernel_params=self._kernel_params_cache,
            deterministic_objective=self._deterministic,
        )
        self._kernel_params_cache = kernel_params

        acqf_params = acqf.create_acqf_params(
            acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=normalized_params,
            Y=standarized_score_vals,
        )

        normalized_param = self._optimize_acqf(
            acqf_params, normalized_params[np.argmax(standarized_score_vals), :]
        )
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)