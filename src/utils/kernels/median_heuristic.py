import torch
from torch import Tensor


def compute_median_heuristic_lengthscales(X: torch.FloatTensor) -> Tensor:
    num_features = X.shape[1]
    median_heuristics = torch.zeros(num_features)
    for feature in range(num_features):
        median_heuristic = torch.median(torch.cdist(X[:, [feature]], X[:, [feature]]))
        if median_heuristic != 0:
            median_heuristics[feature] = median_heuristic
        else:
            median_heuristics[feature] = torch.tensor(1.0)

    return median_heuristics
