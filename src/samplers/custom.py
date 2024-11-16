import optuna
from optuna.samplers import QMCSampler, RandomSampler


class CustomRandomSampler(RandomSampler):
    def __init__(self, seed=None, unique_sampling: bool = False):
        super().__init__(seed=seed)
        self.unique_sampling = unique_sampling  # Flag to enable/disable unique sampling

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Retrieve all previously evaluated parameter points
        evaluated_points = [
            trial.params
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        while True:
            sample = super().sample_independent(
                study, trial, param_name, param_distribution
            )

            if self.unique_sampling:
                # If unique_sampling is enabled, avoid duplicating points
                if all([sample != params[param_name] for params in evaluated_points]):
                    return sample
            else:
                # If unique_sampling is disabled, return the sample directly
                return sample


class CustomQMCSampler(QMCSampler):
    def __init__(
        self,
        seed=None,
        qmc_type="sobol",
        scramble=True,
        independent_sampler=None,
        unique_sampling: bool = False
    ):
        super().__init__(
            seed=seed,
            qmc_type=qmc_type,
            scramble=scramble,
            independent_sampler=independent_sampler,
        )
        self.unique_sampling = unique_sampling  # Flag to enable/disable unique sampling

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Retrieve all previously evaluated parameter points
        evaluated_points = [
            trial.params
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        while True:
            sample = super().sample_independent(
                study, trial, param_name, param_distribution
            )

            if self.unique_sampling:
                # If unique_sampling is enabled, avoid duplicating points
                if all([sample != params[param_name] for params in evaluated_points]):
                    return sample
            else:
                # If unique_sampling is disabled, return the sample directly
                return sample
