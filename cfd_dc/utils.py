import torch


def custom_global_avg_pool(features_list):
    """
    Handles different feature dimensions by averaging each feature vector
    to a single scalar.

    Args:
        features_list (list of torch.Tensor): A list of N tensors,
                                 each of shape [batch_size, features_i]

    Returns:
        torch.Tensor: A tensor of shape [batch_size, N]
    """

    batch_size = features_list[0].size(0)
    num_channels = len(features_list)
    avg_pool_result = torch.zeros(batch_size, num_channels, device=features_list[0].device)

    for idx, features in enumerate(features_list):
        # Compute the mean along the features dimension
        channel_mean = features.mean(dim=1)  # [batch_size]
        avg_pool_result[:, idx] = channel_mean

    normalized_avgpool_result = min_max_normalization(avg_pool_result)

    return normalized_avgpool_result


def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    if max_val - min_val == 0:
        return tensor - min_val
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized