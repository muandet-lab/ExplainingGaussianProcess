import matplotlib.pylab as plt
import pandas as pd
import shap
import torch
from torch import FloatTensor


def summary_plot(shapley_values: torch.Tensor,
                 query_data: torch.Tensor,
                 **kwargs
                 ):
    return shap.summary_plot(shapley_values.T.numpy(), query_data.numpy(), show=False,
                             **kwargs)


def global_importance_bar_plot(mean_shapley_values: FloatTensor, feature_names: list[str], stds: FloatTensor,
                               topk: int = 10):
    global_explanations = pd.DataFrame(mean_shapley_values, index=feature_names, columns=["mean_svs"])
    global_explanations["stds"] = stds
    global_explanations.sort_values("mean_svs", ascending=True, inplace=True)

    global_explanations = global_explanations.iloc[-topk:, :]

    plt.barh(global_explanations.index, global_explanations["mean_svs"], alpha=0.5)
    plt.errorbar(global_explanations["mean_svs"], global_explanations.index, xerr=global_explanations.stds, ls="none")
    plt.title("Global contribution with 1 sd error bar")
    plt.ylabel("Global Feature contribution")

    return None
