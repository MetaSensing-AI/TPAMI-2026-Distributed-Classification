import torch
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models import MobileNetV2
import os


class MultiViewDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        samples = [torch.tensor(view, dtype=torch.float) for view in self.data[idx]]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return samples, label


# --- Underwater Acoustic Data Processing  ---
def PANNs_MobileNet(X, panns_model, device):
    x_embeddings = torch.zeros(X.shape[0], X.shape[1], 1024)
    x = torch.tensor(X).to(torch.float32).to(device)

    num_nodes = X.shape[1]

    for i in range(num_nodes):
        for j in range(X.shape[0]):
            panns_model.eval()
            with torch.no_grad():
                uu = x[j, i, :]
                uu = torch.unsqueeze(uu, dim=0)
                x_panns = panns_model(uu)
                x_panns = x_panns['embedding']
                x_embeddings[j, i, :] = x_panns
    return x_embeddings


def get_underwater_loaders(data_root, batch_size=32, device='cpu',
                           panns_path='pretrained_model/MobileNetV2_mAP=0.383.pth'):
    """
    Load and process underwater acoustic data
    """
    X_list_path = os.path.join(data_root, 'underwater_embedding_20.pt')
    Y_list_path = os.path.join(data_root, 'underwater_y.pt')

    print("Loading pre-extracted features by PANNs_MobileNet...")
    X_list = torch.load(X_list_path)
    Y_list = torch.load(Y_list_path)

    class_counts = Counter(Y_list)
    num_classes = len(class_counts)
    print(f"Loaded {len(Y_list)} samples across {num_classes} classes.")

    num_features_per_view = [view.shape[1] for view in X_list]
    num_nodes = len(X_list)
    print(f"Number of nodes: {num_nodes}")

    scaled_X_list = []
    for X in X_list:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler.fit_transform(X)
        scaled_X_list.append(scaled_X)

    samples_list = [None] * len(X_list[0])
    for i in range(len(samples_list)):
        samples_list[i] = [view[i, :] for view in scaled_X_list]

    # 70% Train, 15% Validation, 15% Test
    train_val_samples, test_samples, y_train_val, y_test = train_test_split(
        samples_list, Y_list, test_size=0.15, stratify=Y_list, random_state=42
    )
    # 0.15 / 0.85 = 0.1765
    train_samples, val_samples, y_train, y_val = train_test_split(
        train_val_samples, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    # count_label(y_train, y_val, y_test, Y_list)

    train_dataset = MultiViewDataset(train_samples, y_train)
    val_dataset = MultiViewDataset(val_samples, y_val)
    test_dataset = MultiViewDataset(test_samples, y_test)

    tr_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, num_features_per_view, num_nodes, num_classes


# --- Multi-View Data Loading ---
def get_multiview_loaders(dataset_name, data_root, batch_size=32):
    print(f"Loading multi-view dataset: {dataset_name}...")

    if dataset_name == 'ALOI-100':
        data_path = os.path.join(data_root, 'ALOI_100.mat')
        data = scio.loadmat(data_path)
        X_list = data['fea'][0].squeeze().tolist()
        Y_list0 = data['gt'].squeeze().tolist()
        Y_list = [item - 1 for item in Y_list0]
        num_nodes = 4

    elif dataset_name == 'ALOI-100-8':
        data_path = os.path.join(data_root, 'ALOI_100_8v.mat')
        data = scio.loadmat(data_path)
        struct_array = data['ALOI_100_8v']
        X_list_raw = struct_array['fea'][0].squeeze().tolist()
        Y_list_raw = struct_array['gt'].squeeze().tolist()
        Y_list = Y_list_raw.flatten()

        X_list = []
        for view_matrix in X_list_raw[0]:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_X = scaler.fit_transform(view_matrix)
            X_list.append(scaled_X)
        num_nodes = 8

    elif dataset_name == 'Caltech101-7':
        data_path = os.path.join(data_root, 'Caltech101-7.mat')
        data = scio.loadmat(data_path)
        X_list_raw = data['X'][0].squeeze().tolist()
        Y_list_raw = data['Y'].squeeze().tolist()
        Y_list = [item - 1 for item in Y_list_raw]  # Labels start from 0

        X_list = []
        for X in X_list_raw:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_X = scaler.fit_transform(X)
            X_list.append(scaled_X)
        num_nodes = 6

    elif dataset_name == 'Handwritten':
        data_path = os.path.join(data_root, 'handwritten.mat')
        data = scio.loadmat(data_path)
        X_list_raw = data['X'][0].squeeze().tolist()
        Y_list = data['Y'].squeeze().tolist()  # Labels are already 0-9

        # Handwritten 需要归一化
        X_list = []
        for X in X_list_raw:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_X = scaler.fit_transform(X)
            X_list.append(scaled_X)
        num_nodes = 6

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    class_counts = Counter(Y_list)
    num_classes = len(class_counts)
    print(f"Loaded {len(Y_list)} samples across {num_classes} classes.")

    num_features_per_view = [view.shape[1] for view in X_list]
    print(f"Number of nodes: {num_nodes}")
    print(f"Features per view: {num_features_per_view}")

    samples_list = [None] * len(X_list[0])
    for i in range(len(samples_list)):
        samples_list[i] = [view[i, :] for view in X_list]

    # 70 % Train, 15 % Validation, 15 % Test
    train_val_samples, test_samples, y_train_val, y_test = train_test_split(
        samples_list, Y_list, test_size=0.15, stratify=Y_list, random_state=42
    )
    # 0.15 / 0.85 = 0.1765
    train_samples, val_samples, y_train, y_val = train_test_split(
        train_val_samples, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
    )

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    # count_label(y_train, y_val, y_test, Y_list)

    train_dataset = MultiViewDataset(train_samples, y_train)
    val_dataset = MultiViewDataset(val_samples, y_val)
    test_dataset = MultiViewDataset(test_samples, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_features_per_view, num_nodes, num_classes