import torch
import torch.nn.functional as F


def similarity_matrix(a, b, eps=1e-8):
    return F.normalize(a, dim=1, eps=eps) @ F.normalize(b, dim=1, eps=eps).T


def nearest_neighbor_indices(key, candidates, num_neighbors=1):
    if num_neighbors < 1 or num_neighbors > candidates.shape[0]:
        raise ValueError("num_neighbors must be between 1 and current source batch size")
    return torch.topk(similarity_matrix(key, candidates), k=num_neighbors, dim=1).indices


def nearest_neighbors(key, candidates, num_neighbors=1, return_indices=False):
    indices = nearest_neighbor_indices(key, candidates, num_neighbors)
    neighbors = torch.stack([candidates[indices[:, i]] for i in range(num_neighbors)])
    return (neighbors, indices) if return_indices else neighbors
