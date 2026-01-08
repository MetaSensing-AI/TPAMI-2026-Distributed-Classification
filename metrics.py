import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def eachnode_correct(all_node_logits, labels):
    """
    Calculate the number of correctly classified samples for each node in the current batch
    Args:
        all_node_logits (Tensor): [batch_size, num_nodes, num_classes]
        labels (Tensor): [batch_size]
    Returns:
        list: length num_nodes, contains the correct count for each node
    """
    num_nodes = all_node_logits.shape[1]
    each_current_correct = []
    for dd in range(num_nodes):
        currentnode_logits = all_node_logits[:, dd, :]
        currentnode_correct = (currentnode_logits.argmax(1) == labels).float().sum().item()
        each_current_correct.append(currentnode_correct)
    return each_current_correct


def eachnode_loss_add(all_node_logits, labels, loss_func, device):
    """
    Calculate the total loss for all nodes
    """
    loss_sum = 0
    num_nodes = all_node_logits.shape[1]
    for dd in range(num_nodes):
        currentnode_logits = all_node_logits[:, dd, :]
        currentnode_loss = loss_func(currentnode_logits.to(device), labels.to(device, dtype=torch.long))
        loss_sum += currentnode_loss
    return loss_sum


def acc_F1_distributed(pernode_allbatch_corr, all_labels, all_node_logits_allbatch, num_nodes):
    """
    Calculate Mean/Max/Min ACC and F1 for all nodes across the entire dataset
    Args:
        pernode_allbatch_corr: Correct counts for each node (inner list) in each batch (outer list)
        all_labels: All ground truth labels
        all_node_logits_allbatch: Logits for each batch [batch_size, num_nodes, num_classes]
        num_nodes: Number of nodes
    Returns:
        acc_mean, acc_max, acc_min, F1_mean, F1_max, F1_min, per_node_acc, per_node_f1
    """
    num_samples_allbatch = len(all_labels)

    tensor_allbatch_corr = torch.tensor(pernode_allbatch_corr)

    sums_per_node = tensor_allbatch_corr.sum(dim=0)

    # per_node_acc: [num_nodes], accuracy for each node
    per_node_acc = np.array(sums_per_node) / num_samples_allbatch * 100
    acc_mean = torch.mean(torch.tensor(per_node_acc))
    acc_max = torch.max(torch.tensor(per_node_acc))
    acc_min = torch.min(torch.tensor(per_node_acc))

    all_labels_tensor = torch.tensor(all_labels)
    allbatch_per_logits = torch.cat(all_node_logits_allbatch, dim=0)

    per_node_f1 = []
    for ii in range(num_nodes):
        currentnode_logits = allbatch_per_logits[:, ii, :]
        _, predicted_classes = torch.max(currentnode_logits, dim=1)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            all_labels_tensor.cpu(), predicted_classes.cpu(), average='macro', zero_division=0
        )
        per_node_f1.append(f1_score * 100)

    F1_mean = torch.mean(torch.tensor(per_node_f1))
    F1_max = torch.max(torch.tensor(per_node_f1))
    F1_min = torch.min(torch.tensor(per_node_f1))

    return acc_mean, acc_max, acc_min, F1_mean, F1_max, F1_min, per_node_acc, per_node_f1