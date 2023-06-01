from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import FloatTensor, Tensor

from src.explanation_algorithms.RKHSSHAP import RKHSSHAP


@dataclass(kw_only=True)
class GPSHAP(RKHSSHAP):
    X_explained: FloatTensor = field(init=False)
    mean_stochastic_shapley_values: Tensor = field(init=False)
    low_rank_component_of_covariance: Tensor = field(init=False)

    def __post_init__(self):
        super(GPSHAP, self).__post_init__()

    def fit_gpshap(self, X: FloatTensor, num_coalitions: int) -> None:
        self.X_explained = X
        self.fit_rkhsshap(X, num_coalitions=num_coalitions)

        self.mean_stochastic_shapley_values = self.return_deterministic_shapley_values()
        self.low_rank_component_of_covariance = torch.einsum(
            "ij,jkl->ikl", self._compute_kernelSHAP_projection_matrix(),
            self._compute_tensor_mode_product_of_cmps_with_choleksy_of_posterior_covariance())

    def return_mean_stochastic_shapley_values(self) -> Tensor:
        return self.mean_stochastic_shapley_values

    def return_gpshap_uncertainties_across_all_queries(self):
        """ build a tensor of 4 dimension encapsulating covariances across both features and observations.
            Size: [num_features, num_features, num_queries, num_queries]
        """
        Psi = self.low_rank_component_of_covariance
        return torch.einsum("ijk,lmn->imkn", Psi, Psi.transpose(0, 1)) * self.scale ** 2

    def return_gpshap_uncertainties_for_each_query(self):
        """ build a tensor of 3 dimension encapsulating covariances between features for each observation.
            Size: [num_features, num_features, num_queries, num_queries]
        """
        Psi = self.low_rank_component_of_covariance
        return torch.einsum("ijk,lmk->imk", Psi, Psi.transpose(0, 1)) * self.scale ** 2

    def return_gpshap_uncertainties_for_query_i_j(self, i: int, j: int):
        """ compute the cross covariance matrix for observation i and j.
            Size: [num_features, num_features]
        """
        Psi = self.low_rank_component_of_covariance
        return (Psi[:, :, i] @ Psi[:, :, j].T) * self.scale ** 2

    def _compute_kernelSHAP_projection_matrix(self) -> Tensor:
        """compute the matrix projection matrix (ZtWZ)^{-1}(ZtW)"""
        ZtW = self.coalitions.t() @ torch.diag(self.weights.squeeze())
        ZtWZ = ZtW @ self.coalitions
        return torch.cholesky_solve(ZtW, torch.linalg.cholesky(ZtWZ))

    def _compute_tensor_mode_product_of_cmps_with_choleksy_of_posterior_covariance(self) -> Tensor:
        """ compute the tensor mode product of the conditional mean projections with the cholesky decomposition of
            posterior covariance

        Returns
        -------
        a tensor of the shape [num_coalitions, num_inducing_points, num_queries]
        """
        conditional_mean_projections = self.conditional_mean_projections
        zeros = torch.zeros(1, conditional_mean_projections.shape[1], conditional_mean_projections.shape[2])
        conditional_mean_projections = torch.cat([zeros, conditional_mean_projections], dim=0)

        return torch.einsum(
            "ijk,jl->ilk", conditional_mean_projections, self.posterior_cov_of_inducing_data.cholesky())
