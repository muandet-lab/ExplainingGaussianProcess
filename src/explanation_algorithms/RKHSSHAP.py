from __future__ import annotations

from dataclasses import dataclass, field

import torch
from gpytorch.kernels import Kernel
from joblib import Parallel, delayed
from torch import FloatTensor, BoolTensor, Tensor
from tqdm import tqdm

from src.gp_models.ExactGPRegression import ExactGPRegression
from src.gp_models.VariationalGPRegression import VariationalGPRegression
from src.utils.shapley_procedure.preparing_weights_and_coalitions import compute_weights_and_coalitions


@dataclass(kw_only=True)
class RKHSSHAP(object):
    """Run the RKHS-SHAP algorithm to explain the posterior mean (equivalent to a kernel ridge regression)
     of a Gaussian process model.

    """
    train_X: FloatTensor
    model: ExactGPRegression | VariationalGPRegression
    kernel: Kernel
    include_likelihood_noise_for_explanation: bool
    scale: FloatTensor  # scale of labels

    mean_stochastic_value_function_evaluations: Tensor = field(init=False)
    conditional_mean_projections: FloatTensor | Tensor = field(init=False)
    coalitions: BoolTensor = field(init=False)
    weights: FloatTensor = field(init=False)

    cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
    num_cpus: int = field(init=False, default=6)

    def __post_init__(self):
        self.kernel_lengthscales = self.model.lengthscale

        if self.model.num_inducing_points is not None:
            self.inducing_points = self._scaled_by_lengthscales(self.model.inducing_points)
        else:
            self.inducing_points = self._scaled_by_lengthscales(self.train_X)

        mean, cov = self.model.compute_posterior_mean_and_covariance_of_data(
            data=self.inducing_points * self.kernel_lengthscales,
            likelihood=self.include_likelihood_noise_for_explanation
        )
        self.posterior_mean_of_inducing_data = mean.detach()
        self.posterior_cov_of_inducing_data = cov.detach()

    def fit_rkhsshap(self, X: FloatTensor, num_coalitions: int = 100) -> None:

        X = self._scaled_by_lengthscales(X)
        self.weights, self.coalitions = compute_weights_and_coalitions(num_features=X.shape[1],
                                                                       num_coalitions=num_coalitions)
        self.conditional_mean_projections = self._compute_conditional_mean_projections(X)
        self.mean_stochastic_value_function_evaluations = torch.concat([
            self.posterior_mean_of_inducing_data.mean() * torch.ones((1, X.shape[0])),
            torch.einsum(
                'ijk,j->ik', self.conditional_mean_projections, self.posterior_mean_of_inducing_data
            )
        ])

    def return_deterministic_shapley_values(self) -> FloatTensor:
        return _solve_weighted_least_square_regression(kernelSHAP_weights=self.weights,
                                                       coalitions=self.coalitions,
                                                       regression_target=self.mean_stochastic_value_function_evaluations
                                                       ) * self.scale

    def _compute_conditional_mean_projections(self, X):

        #  Compute the conditional mean projections using joblib parallelisation.
        minus_first_coalitions = self.coalitions[1:]  # remove the first row of 0s.
        compute_conditional_mean_projections = lambda S: self._compute_conditional_mean_projection(S.bool(), X)
        return torch.stack(
            Parallel(n_jobs=self.num_cpus)(
                delayed(compute_conditional_mean_projections)(S.bool())
                for S in tqdm(minus_first_coalitions)
            )
        )

    def _compute_value_function_at_coalition(self, S: BoolTensor, X: FloatTensor):
        """compute the value function E[f(X) | X_S=x_S]

        Parameters
        ----------
        X: size = [num_data x num_features]
        S: binary vector of coalition

        Returns
        -------
        the conditional mean
        """
        if S.sum() == 0:  # no active feature
            return (torch.ones((1, X.shape[0])) * self.posterior_mean_of_inducing_data.mean()).squeeze()

        conditional_mean_projection = self._compute_conditional_mean_projection(S, X)

        return conditional_mean_projection.T @ self.posterior_mean_of_inducing_data

    def _compute_conditional_mean_projection(self, S: BoolTensor, X: FloatTensor):
        """ compute the expression k_S(x, X)(K_SS + lambda I)^{-1} that can be reused multiple times
        """
        k_inducingXS_XS = self.kernel(self.inducing_points[:, S], X[:, S])
        return (self.kernel(self.inducing_points[:, S])).add_diag(
            self.model.num_inducing_points * self.cme_regularisation).inv_matmul(
            k_inducingXS_XS.evaluate()).detach()

    def _scaled_by_lengthscales(self, X: torch.FloatTensor) -> FloatTensor:
        return X / self.kernel_lengthscales


def _solve_weighted_least_square_regression(kernelSHAP_weights: FloatTensor,
                                            coalitions: BoolTensor,
                                            regression_target: FloatTensor | Tensor,
                                            ) -> FloatTensor:
    weighted_regression_target = regression_target * kernelSHAP_weights
    ZtWvx = coalitions.t() @ weighted_regression_target
    L = torch.linalg.cholesky(coalitions.t() @ (coalitions * kernelSHAP_weights))

    return torch.cholesky_solve(ZtWvx, L).detach()
