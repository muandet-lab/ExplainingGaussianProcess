import torch

from src.utils.shapley_procedure.coalitions import build_coalitions
from src.utils.shapley_procedure.weights import compute_kernelshap_weights_for_pre_coalitions


def compute_weights_and_coalitions(num_features: int, num_coalitions: int) -> list[
    torch.tensor, torch.tensor]:
    """compute weights of KernelSHAP formulation and build coalitions
    """
    # sample coalitions, edge cases are not considered yet.
    pre_coalitions = torch.from_numpy(
        build_coalitions(num_features=num_features, num_coalitions=num_coalitions)
    ).bool()

    weights = compute_kernelshap_weights_for_pre_coalitions(num_features, pre_coalitions)

    bool = torch.isfinite(weights.squeeze())
    edge_case_weights = torch.tensor([1e3]).float()
    non_edge_case_weights = weights[bool]
    non_edge_case_weights = non_edge_case_weights / non_edge_case_weights.sum()
    non_edge_case_coalitions = pre_coalitions[bool, :]

    return [torch.concat([edge_case_weights,
                          non_edge_case_weights,
                          edge_case_weights],
                         dim=0).unsqueeze(-1),
            torch.concat([torch.zeros(1, num_features),
                          non_edge_case_coalitions,
                          torch.ones(1, num_features)],
                         dim=0)
            ]  # weights and coalitions
