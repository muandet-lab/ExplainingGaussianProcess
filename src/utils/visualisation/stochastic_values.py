import matplotlib.pylab as plt
import torch

from src.explanation_algorithms.BayesGPSHAP import BayesGPSHAP


def local_explanation_plot(data_id: int,
                           feature_names: list[str],
                           explanation_algorithm: BayesGPSHAP,
                           uncertainty_source: str,
                           coverage: float = 1.0):
    X = explanation_algorithm.X_explained
    mean_stochastic_shapley_values = explanation_algorithm.return_mean_stochastic_shapley_values()

    if uncertainty_source == "GPSHAP":
        uncertainties = explanation_algorithm.return_gpshap_uncertainties_for_each_query()[:, :, data_id]
    elif uncertainty_source == "BayesSHAP":
        uncertainties = explanation_algorithm.return_bayes_shap_uncertainties(X)[:, :, data_id]
    elif uncertainty_source == "BayesGPSHAP":
        uncertainties = explanation_algorithm.return_bayes_gpshap_uncertainties(X)[:, :, data_id]

    uncertainties = coverage * torch.diag(uncertainties).sqrt()

    plt.bar(feature_names, mean_stochastic_shapley_values[:, data_id], alpha=0.5)
    plt.errorbar(feature_names, mean_stochastic_shapley_values[:, data_id], yerr=uncertainties, capsize=3, ls="None")
    plt.title(f"Stochastic Explanations for data {data_id} ({uncertainty_source})")
    plt.xlabel("feature names")
    plt.ylabel("stochastic Shapley values")
    plt.show()
