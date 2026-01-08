import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import argparse
import os
import numpy as np

from cfd_dc.models import CFD_DC, CFD_DC_50Compress, CFD_DC_NoWeighting
from data_loader import get_multiview_loaders
from metrics import eachnode_correct, eachnode_loss_add, acc_F1_distributed
from train_utils import EarlyStopping, draw_fig


def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tr_loader, va_loader, te_loader, num_features, num_nodes, num_classes = get_multiview_loaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size
    )

    if args.compress_mode == 'fixed':
        print(f"Initializing CFD-DC model with fixed d_g = {args.dg}...")
        model = CFD_DC_NoWeighting(
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_features_per_view=num_features,
            fea_out=args.dg,
            d_prob=args.dropout
        )
    elif args.compress_mode == 'relative':
        print("Initializing CFD-DC model with relative d_g = d_f / 2...")
        model = CFD_DC_50Compress(
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_features_per_view=num_features,
            d_prob=args.dropout
        )
    else:
        raise ValueError("Invalid compress_mode. Choose 'fixed' or 'relative'.")

    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True)
    history = defaultdict(list)
    best_val_acc = 0.0

    print(f"--- Starting Training for {args.epochs} epochs ---")

    for epoch in range(1, args.epochs + 1):

        model.train()
        tr_pernode_allbatch_corr = []
        all_tr_node_logits = []
        all_tr_labels = []
        total_tr_loss = 0

        for tr_feature, tr_labels in tr_loader:
            tr_labels = tr_labels.to(device)
            tr_all_node_logits = model(tr_feature)  # [B, N, C]

            tr_each_current_correct = eachnode_correct(tr_all_node_logits, tr_labels)
            tr_pernode_allbatch_corr.append(tr_each_current_correct)
            all_tr_node_logits.append(tr_all_node_logits.detach())
            all_tr_labels.extend(tr_labels.cpu().numpy())

            tr_loss = eachnode_loss_add(tr_all_node_logits, tr_labels, loss_func, device)
            total_tr_loss += tr_loss.item()

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        model.eval()
        val_pernode_allbatch_corr = []
        all_val_node_logits = []
        all_val_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for val_feature, val_labels in va_loader:
                val_labels = val_labels.to(device)
                val_all_node_logits = model(val_feature)  # [B, N, C]

                val_each_current_correct = eachnode_correct(val_all_node_logits, val_labels)
                val_pernode_allbatch_corr.append(val_each_current_correct)
                all_val_node_logits.append(val_all_node_logits.detach())
                all_val_labels.extend(val_labels.cpu().numpy())

                val_loss = eachnode_loss_add(val_all_node_logits, val_labels, loss_func, device)
                total_val_loss += val_loss.item()

        tr_acc_mean, _, _, tr_F1_mean, _, _, _, _ = acc_F1_distributed(
            tr_pernode_allbatch_corr, all_tr_labels, all_tr_node_logits, num_nodes
        )
        val_acc_mean, _, _, val_F1_mean, _, _, _, _ = acc_F1_distributed(
            val_pernode_allbatch_corr, all_val_labels, all_val_node_logits, num_nodes
        )
        tr_loss_epoch = total_tr_loss / len(tr_loader.dataset)
        val_loss_epoch = total_val_loss / len(va_loader.dataset)

        history['train_acc'].append(tr_acc_mean)
        history['train_loss'].append(tr_loss_epoch)
        history['val_acc'].append(val_acc_mean)
        history['val_loss'].append(val_loss_epoch)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {tr_loss_epoch:.4f}, Train Acc: {tr_acc_mean:.2f}%, "
              f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_mean:.2f}%")

        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            torch.save(model.state_dict(), f"best_model_{args.dataset}_{args.compress_mode}_{args.dg}.pkl")
            print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

        # early_stopper(val_acc_mean)
        # if early_stopper.early_stop:
        #     break

    # --- Testing ---
    print("--- Starting Testing ---")
    model.load_state_dict(torch.load(f"best_model_{args.dataset}_{args.compress_mode}_{args.dg}.pkl"))
    model.eval()

    te_pernode_allbatch_corr = []
    all_te_node_logits = []
    all_te_labels = []

    with torch.no_grad():
        for te_feature, te_labels in te_loader:
            te_labels = te_labels.to(device)
            te_all_node_logits = model(te_feature)  # [B, N, C]

            te_each_current_correct = eachnode_correct(te_all_node_logits, te_labels)
            te_pernode_allbatch_corr.append(te_each_current_correct)
            all_te_node_logits.append(te_all_node_logits.detach())
            all_te_labels.extend(te_labels.cpu().numpy())

    te_acc_mean, te_acc_max, te_acc_min, te_F1_mean, te_F1_max, te_F1_min, te_per_node_acc, te_per_node_f1 = acc_F1_distributed(
        te_pernode_allbatch_corr, all_te_labels, all_te_node_logits, num_nodes
    )

    print("\n--- Results for the Single Run ---")
    print(f"Mean Accuracy: {te_acc_mean:.2f}%")
    print(f"Mean Macro F1-Score: {te_F1_mean:.2f}%")
    print(f"Best-Node Accuracy: {te_acc_max:.2f}%")
    print(f"Best-Node Macro F1-Score: {te_F1_max:.2f}%")
    print(f"Worst-Node Accuracy: {te_acc_min:.2f}%")
    print(f"Worst-Node Macro F1-Score: {te_F1_min:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Multi-View Classification Experiments")
    parser.add_argument('--dataset', type=str, default='Handwritten',
                        choices=['ALOI-100', 'ALOI-100-8', 'Caltech101-7', 'Handwritten'],
                        help='Name of the dataset to run.')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Directory where the .mat files are stored.')
    parser.add_argument('--compress_mode', type=str, default='fixed', choices=['fixed', 'relative'],
                        help="Compression mode: 'fixed' (d_g=8) or 'relative' (d_g=d_f/2)")
    parser.add_argument('--dg', type=int, default=8,
                        help='Fixed compressed feature dimension (d_g). Used only if compress_mode="fixed".')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()

    run_experiment(args)