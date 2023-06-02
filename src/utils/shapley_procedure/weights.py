import operator as op
from functools import reduce

import torch
from torch import BoolTensor, FloatTensor


def nCk(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def compute_kernelshap_weights_for_pre_coalitions(num_features: int, pre_coalitions: BoolTensor) -> FloatTensor:
    const = torch.lgamma(torch.tensor(num_features) + 1)
    abs_Ss = pre_coalitions.sum(axis=1)

    # sterling's approximation for factorials
    if num_features >= 14:
        num_features_choose_S = torch.exp(const - torch.lgamma((num_features - abs_Ss) + 1)) - torch.lgamma(abs_Ss + 1)
    else:
        num_features_choose_S = torch.tensor([nCk(num_features, abs_Ss[i]) for i in range(len(abs_Ss))])

    return (num_features - 1) / (num_features_choose_S * (abs_Ss) * (num_features - abs_Ss))
