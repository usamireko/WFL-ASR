import torch
import numpy as np

def pad_or_trim(tensor, target_len):
    current_len = tensor.shape[-1]
    if current_len == target_len:
        return tensor
    elif current_len < target_len:
        pad_width = target_len - current_len
        return torch.nn.functional.pad(tensor, (0, pad_width))
    else:
        return tensor[..., :target_len]

def median_cluster_filter(indices, min_distance=2):
    if not indices:
        return []

    filtered = []
    cluster = [indices[0]]

    for idx in indices[1:]:
        if idx - cluster[-1] <= min_distance:
            cluster.append(idx)
        else:
            filtered.append(int(np.median(cluster)))
            cluster = [idx]

    if cluster:
        filtered.append(int(np.median(cluster)))

    return filtered