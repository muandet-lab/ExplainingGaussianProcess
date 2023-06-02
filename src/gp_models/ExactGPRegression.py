from dataclasses import dataclass, field
from typing import Optional

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import InducingPointKernel, Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import FloatTensor
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer

from src.utils.kernels.inducing_points import compute_inducing_points
from src.utils.kernels.median_heuristic import compute_median_heuristic_lengthscales


class ExactGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: FloatTensor, train_y: FloatTensor, kernel: Kernel,
                 likelihood: GaussianLikelihood) -> None:
        super(ExactGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covariance_module = kernel(ard_num_dims=train_x.shape[1])
        self.covariance_module.lengthscale = compute_median_heuristic_lengthscales(train_x)

    def forward(self, x: FloatTensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x),
                                  self.covariance_module(x))


class SparseGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self,
                 train_x: FloatTensor,
                 train_y: FloatTensor,
                 kernel: Kernel,
                 inducing_x: FloatTensor,
                 likelihood: GaussianLikelihood,
                 ):
        super(SparseGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = kernel(ard_num_dims=train_x.shape[1])
        self.covariance_module = InducingPointKernel(self.base_covar_module,
                                                     inducing_points=inducing_x,
                                                     likelihood=likelihood)
        self.base_covar_module.lengthscale = compute_median_heuristic_lengthscales(inducing_x)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x),
                                  self.covariance_module(x))


@dataclass()
class ExactGPRegression(object):
    train_X: FloatTensor
    train_y: FloatTensor
    kernel: Kernel
    num_inducing_points: Optional[int] = field(default=None)

    lengthscale: FloatTensor = field(init=False, default=None)
    optimizer: Optimizer = field(init=False, default=None)
    marginal_log_likelihood: MarginalLogLikelihood = field(init=False, default=None)

    def __post_init__(self):
        self.likelihood = GaussianLikelihood()
        if self.num_inducing_points is not None:
            self.inducing_points = compute_inducing_points(self.train_X, num_inducing_points=self.num_inducing_points)
            self.model = SparseGPRegressionModel(train_x=self.train_X,
                                                 train_y=self.train_y,
                                                 kernel=self.kernel,
                                                 inducing_x=self.inducing_points,
                                                 likelihood=self.likelihood
                                                 )
        else:
            self.model = ExactGPRegressionModel(self.train_X, self.train_y, self.kernel, self.likelihood)

        # enter train mode
        self.model.train()
        self.likelihood.train()

    def fit(self, learning_rate: float = 1e-2, training_iteration: int = 500, verbose: bool = False) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        self.marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.likelihood,
                                                                                model=self.model)

        for rd in range(training_iteration):
            self.optimizer.zero_grad()
            prediction = self.model(self.train_X)
            loss = -self.marginal_log_likelihood(prediction, self.train_y)
            loss.backward()

            if verbose:
                if rd % int(training_iteration / 5) == 0:
                    print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                        rd + 1, training_iteration, loss.item(),
                        self.model.likelihood.noise.item()
                    ))

            self.optimizer.step()

        if self.num_inducing_points is not None:
            self.lengthscale = self.model.base_covar_module.lengthscale.detach()
        else:
            self.lengthscale = self.model.covariance_module.lengthscale.detach()

    def predict(self, test_X: FloatTensor) -> Distribution:
        self.model.eval()
        self.likelihood.eval()

        return self.likelihood(self.model(test_X))

    def compute_posterior_mean_and_covariance_of_data(self, data: FloatTensor):
        predictive_distribution = self.predict(data)
        return predictive_distribution.mean.detach(), predictive_distribution.covariance_matrix.detach()

    def compute_posterior_mean_and_covariance_of_training_data(self):
        return self.compute_posterior_mean_and_covariance_of_data(data=self.train_X)
