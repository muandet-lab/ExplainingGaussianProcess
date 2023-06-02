import numpy as np


def build_coalitions(num_features: int, num_coalitions: int) -> np.ndarray:
    max_range = min(2 ** num_features, 2 ** 63 - 1)

    if num_coalitions >= max_range:
        configs = np.arange(max_range)
        return _generate_coalitions_from_integers(configs, num_features)

    return _sample_coalitions_from_binomial(num_coalitions=num_coalitions, num_features=num_features)


def _generate_coalitions_from_integers(indices: np.array, num_features: int = 10) -> np.ndarray:
    Z = np.zeros((indices.shape[0], num_features))
    rest = indices
    valid_rows = rest > 0
    while True:
        set_to_1 = np.floor(np.log2(rest)).astype(int)
        set_to_1_prime = set_to_1[valid_rows][:, np.newaxis]
        p = Z[valid_rows, :]
        np.put_along_axis(p, set_to_1_prime, 1, axis=1)
        Z[valid_rows, :] = p
        rest = rest - 2 ** (np.clip(set_to_1, 0, np.inf))
        valid_rows = rest > 0
        if valid_rows.sum() == 0:
            return Z


def _sample_coalitions_from_binomial(num_coalitions: int, num_features: int) -> np.ndarray:
    """sampling coalitions using binomial distribution and remove duplicates afterward
    """
    Z = np.random.binomial(size=(num_coalitions, num_features), n=1, p=0.5)
    b = 2 ** np.arange(0, num_features)
    unique_ref = (b * Z).sum(axis=1)
    _, idx = np.unique(unique_ref, return_index=True)
    return Z[idx, :]
