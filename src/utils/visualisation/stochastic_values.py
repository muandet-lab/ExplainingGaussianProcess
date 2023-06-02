import matplotlib.pylab as plt
import pandas as pd
import torch

from src.explanation_algorithms.BayesGPSHAP import BayesGPSHAP


def local_explanation_plot(data_id: int,
                           feature_names: list[str],
                           explanation_algorithm: BayesGPSHAP,
                           uncertainty_source: str,
                           coverage: float = 1.0):
    """plot local explanation given specific data id

    :param data_id: the index of the observation you wish to explain
    :param feature_names: list of feature names
    :param explanation_algorithm: the BayesGPSHAP object
    :param uncertainty_source: either "GPSHAP", "BayesSHAP", or "BayesGPSHAP".
    :param coverage: the scalar multiplying the standard deviation. Default to 1 will give you a coverage to 68% of the data.
    """
    X = explanation_algorithm.X_explained
    mean_stochastic_shapley_values = explanation_algorithm.return_mean_stochastic_shapley_values()

    if uncertainty_source == "GPSHAP":
        uncertainties = explanation_algorithm.return_gpshap_uncertainties_for_each_query()[:, :, data_id]
    elif uncertainty_source == "BayesSHAP":
        uncertainties = explanation_algorithm.return_bayes_shap_uncertainties(X)[:, :, data_id]
    elif uncertainty_source == "BayesGPSHAP":
        uncertainties = explanation_algorithm.return_bayes_gpshap_uncertainties(X)[:, :, data_id]

    uncertainties = coverage * torch.diag(uncertainties).sqrt()

    local_importance = pd.DataFrame(mean_stochastic_shapley_values[:, data_id], index=feature_names, columns=["mean"])
    local_importance["std"] = uncertainties
    local_importance = local_importance.sort_values("mean")

    plt.barh(local_importance.index, local_importance["mean"], alpha=0.5)
    plt.errorbar(y=local_importance.index, x=local_importance["mean"], xerr=local_importance["std"], capsize=3,
                 ls="None")
    plt.title(f"Stochastic Explanations for data {data_id} ({uncertainty_source})")
    plt.xlabel("stochastic Shapley values")
    plt.tight_layout()
    plt.show()


def global_explanation_plot(feature_names: list[str],
                            explanation_algorithm: BayesGPSHAP,
                            uncertainty_source: str,
                            coverage: float = 1.0):
    """plot global explanation based on the distribution of the absolute Stochastic Shapley values

        :param data_id: the index of the observation you wish to explain
        :param feature_names: list of feature names
        :param explanation_algorithm: the BayesGPSHAP object
        :param uncertainty_source: either "GPSHAP", "BayesSHAP", or "BayesGPSHAP".
        :param coverage: the scalar multiplying the standard deviation. Default to 1 will give you a coverage to 68% of the data.
        """
    means, stds, cov_mat = explanation_algorithm.compute_global_feature_importances_with_different_uncertainties(
        sample_size=10000, uncertainty_source=uncertainty_source
    )

    global_importance = pd.DataFrame(means, index=feature_names, columns=["mean"])
    global_importance["std"] = stds
    global_importance = global_importance.sort_values("mean")

    plt.barh(global_importance.index, global_importance["mean"], alpha=.5, color="blue")
    plt.errorbar(x=global_importance["mean"],
                 y=global_importance.index,
                 xerr=coverage * global_importance["std"],
                 ls="none",
                 color="blue",
                 capsize=3
                 )
    plt.xlabel("Mean Absolute Stochastic Shapley Values")
    plt.title("Global Explanations")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
