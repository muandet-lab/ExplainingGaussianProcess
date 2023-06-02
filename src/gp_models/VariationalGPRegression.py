from dataclasses import dataclass, field

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import FloatTensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm

from src.utils.kernels.inducing_points import compute_inducing_points
from src.utils.kernels.median_heuristic import compute_median_heuristic_lengthscales


class VariationalGPModel(ApproximateGP):
    def __init__(self, train_X: FloatTensor, kernel: Kernel, num_inducing_points: int):
        self.initial_inducing_points = compute_inducing_points(train_X, num_inducing_points)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing_points)
        variational_strategy = VariationalStrategy(self,
                                                   self.initial_inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True
                                                   )
        super(VariationalGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covariance_module = kernel(ard_num_dims=train_X.shape[1])
        self.covariance_module.lengthscale = compute_median_heuristic_lengthscales(
            self.initial_inducing_points)

    def forward(self, X: FloatTensor) -> MultivariateNormal:
        return MultivariateNormal(mean=self.mean_module(X),
                                  covariance_matrix=self.covariance_module(X))


@dataclass()
class VariationalGPRegression(object):
    train_X: FloatTensor
    train_y: FloatTensor
    kernel: Kernel
    num_inducing_points: int
    batch_size: int

    optimizer: Optimizer = field(init=False, default=None)
    variational_elbo: gpytorch.mlls.VariationalELBO = field(init=False, default=None)
    inducing_points: FloatTensor = field(init=False, default=None)
    lengthscale: FloatTensor = field(init=False, default=None)

    def __post_init__(self):
        self.likelihood = GaussianLikelihood()
        self.model = VariationalGPModel(train_X=self.train_X,
                                        kernel=self.kernel,
                                        num_inducing_points=self.num_inducing_points
                                        )
        self._create_data_loader()

        self.likelihood.train()
        self.model.train()

    def _create_data_loader(self):
        self.train_dataset = TensorDataset(self.train_X, self.train_y)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self, learning_rate: float = 1e-2, training_iteration: int = 500):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=learning_rate)

        self.variational_elbo = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood,
                                                              model=self.model,
                                                              num_data=self.train_X.shape[0]
                                                              )
        epochs_iter = tqdm(range(training_iteration), desc="Epoch")
        for _ in epochs_iter:
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                prediction = self.model(batch_X)
                loss = -self.variational_elbo(prediction, batch_y)
                loss.backward()
                self.optimizer.step()

        self.inducing_points = self.model.variational_strategy.inducing_points.detach()
        self.lengthscale = self.model.covariance_module.lengthscale.detach()

    def predict_include_likelihood(self, test_X: FloatTensor):
        self.model.eval()
        self.likelihood.eval()
        return self.likelihood(self.model(test_X))

    def predict_exclude_likelihood(self, test_X: FloatTensor):
        self.model.eval()
        self.likelihood.eval()
        return self.model(test_X)

    def compute_posterior_mean_and_covariance_of_data(self, data: FloatTensor, likelihood: bool = True):
        if likelihood is True:
            predictive_distribution = self.predict_include_likelihood(data)
        else:
            predictive_distribution = self.predict_exclude_likelihood(data)

        return predictive_distribution.mean.detach(), predictive_distribution.covariance_matrix.detach()
