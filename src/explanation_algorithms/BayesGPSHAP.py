from dataclasses import dataclass, field

import torch
from torch import FloatTensor, Tensor
from torch.distributions.chi2 import Chi2
from torch.distributions.multivariate_normal import MultivariateNormal

from src.explanation_algorithms.GPSHAP import GPSHAP
from src.explanation_algorithms.RKHSSHAP import RKHSSHAP


@dataclass(kw_only=True)
class BayesGPSHAP(GPSHAP, RKHSSHAP):
    bayesSHAP_uncertainties: Tensor = field(init=False)

    def __post_init__(self):
        super(BayesGPSHAP, self).__post_init__()

    def fit_bayesgpshap(self, X: FloatTensor, num_coalitions: int) -> None:
        self.fit_gpshap(X=X, num_coalitions=num_coalitions)
        self.bayesSHAP_uncertainties = self.return_bayes_shap_uncertainties(X)

    def return_bayes_shap_uncertainties(self, X: FloatTensor):
        A_phi_inv = torch.linalg.inv(self.coalitions.t() @ (self.coalitions * self.weights) + torch.eye(X.shape[1]))
        errors = (
                self.mean_stochastic_value_function_evaluations * self.scale - self.coalitions @ self.mean_stochastic_shapley_values)
        weighted_square_errors = torch.diag(errors.T @ (torch.eye(self.weights.shape[0]) * self.weights) @ errors)

        phiTphi = torch.diag(self.mean_stochastic_shapley_values.T @ self.mean_stochastic_shapley_values)
        s_squared = (phiTphi + weighted_square_errors) / self.weights.shape[0]

        chi_squared_distribution = Chi2(df=self.weights.shape[0])
        scaled_inverse_chi_square_samples = s_squared / chi_squared_distribution.sample(sample_shape=[1000, X.shape[0]])
        scaled_inverse_chi_square_averages = scaled_inverse_chi_square_samples.mean(dim=0)

        return torch.stack(
            [A_phi_inv * sample for sample in scaled_inverse_chi_square_averages],
            dim=2
        )

    def return_bayes_gpshap_uncertainties(self, X:FloatTensor):
        gpshap_uncertainties = self.return_gpshap_uncertainties_for_each_query()
        bayesshap_uncertainties = self.return_bayes_shap_uncertainties(X)

        return bayesshap_uncertainties + gpshap_uncertainties

    def compute_global_feature_importances_with_different_uncertainties(self, sample_size: int,
                                                                        uncertainty_source: str):
        """compute the average absolute stochastic shapley values

        Parameters
        ----------
        sample_size: number of samples to take to estimate the moments of folded Gaussians
        uncertainty_source: one of ["GPSHAP", "BayesSHAP", "BayesGPSHAP"]
        """

        num_data, num_features = self.X_explained.shape
        covariance_tensor = self.return_gpshap_uncertainties_across_all_queries()
        mean, std, samples_of_absolute_contributions_ls = [], [], []

        for feature_id in range(num_features):
            covariance_matrix_of_feature_id = covariance_tensor[feature_id, feature_id, :, :]
            mean_vector_of_feature_id = self.mean_stochastic_shapley_values[feature_id, :].unsqueeze(dim=1)
            noise = self.model.likelihood.noise.detach() * self.scale

            if uncertainty_source == "GPSHAP":
                covariance_of_stochastic_sv_of_feature = (
                        covariance_matrix_of_feature_id + noise / num_features * torch.eye(num_data)
                )
            elif uncertainty_source == "BayesSHAP":
                covariance_of_stochastic_sv_of_feature = torch.diag(
                    self.bayesSHAP_uncertainties[feature_id, feature_id, :])
            elif uncertainty_source == "BayesGPSHAP":
                covariance_of_stochastic_sv_of_feature = (
                        covariance_matrix_of_feature_id + noise / num_features * torch.eye(num_data) + torch.diag(
                    self.bayesSHAP_uncertainties[feature_id, feature_id, :])
                )

            stochastic_shapley_values_of_feature_id = MultivariateNormal(
                loc=mean_vector_of_feature_id.squeeze(),
                covariance_matrix=covariance_of_stochastic_sv_of_feature
            )

            samples_of_sv = stochastic_shapley_values_of_feature_id.rsample(sample_shape=[sample_size])
            samples_of_absolution_contributions = samples_of_sv.abs().mean(dim=1)

            samples_of_absolute_contributions_ls.append(samples_of_absolution_contributions)
            mean.append(samples_of_absolution_contributions.mean())
            std.append(samples_of_absolution_contributions.std())

        cov_mat = torch.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(num_features):
                cov_mat[i, j] = ((samples_of_absolute_contributions_ls[i] - samples_of_absolute_contributions_ls[
                    i].mean()) * (samples_of_absolute_contributions_ls[j] - samples_of_absolute_contributions_ls[
                    j].mean())).mean()

        return torch.tensor(mean), torch.tensor(std), cov_mat
