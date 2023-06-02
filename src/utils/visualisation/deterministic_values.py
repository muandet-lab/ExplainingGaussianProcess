import shap
import torch


def summary_plot(shapley_values: torch.Tensor,
                 query_data: torch.Tensor,
                 **kwargs
                 ):
    return shap.summary_plot(shapley_values.T.numpy(), query_data.numpy(), show=False,
                             **kwargs)
