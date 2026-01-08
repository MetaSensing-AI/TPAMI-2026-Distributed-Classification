import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import argparse
import os
import numpy as np
from cfd_dc.models import CFD_DC, CFD_DC_NoWeighting
from data_loader import get_underwater_loaders
from metrics import eachnode_correct, eachnode_loss_add, acc_F1_distributed
from train_utils import EarlyStopping, draw_fig


def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # data_root points to the directory containing .pt and PANNs model .pth files
    tr_loader, va_loader, te_loader, num_features, num_nodes, num_classes = get_underwater_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=device,
        panns_path=args.panns_path
    )

    print(f"Initializing CFD-DC model with d_g = {args.dg}...")
    if args.weighting:
        print("Using SE Weighting Module")
        model = CFD_DC(
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_features_per_view=num_features,
            fea_out=args.dg,
            d_prob=args.dropout
        )
    else:
        print("Using No Weighting Module")
        model = CFD_DC_NoWeighting(
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_features_per_view=num_features,
            fea_out=args.dg,
            d_prob=args.dropout
        )

    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # early_stopper = EarlyStopping(patience=args.patience, verbose=True)
    history = defaultdict(list)
    best_val_acc = 0.0

    # model_save_path = f"best_model_underwater_dg{args.dg}_fail{args.failure_rate}_w{args.weighting}.pkl"
    model_save_path = f"best_model_underwater_dg{args.dg}_w{args.weighting}.pkl"

    print(f"--- Starting Training for {args.epochs} epochs ---")

    for epoch in range(1, args.epochs + 1):

        model.train()
        tr_pernode_allbatch_corr = []
        all_tr_node_logits = []
        all_tr_labels = []
        total_tr_loss = 0

        for tr_feature, tr_labels in tr_loader:
            tr_labels = tr_labels.to(device)

            # malfunctioning sensors that capture only low-energy noise
            node_failure_mask = None
            if args.failure_rate > 0:
                # [batch_size, num_nodes], True = failure
                node_failure_mask = torch.rand(tr_labels.size(0), num_nodes) < args.failure_rate
                node_failure_mask = node_failure_mask.to(device)

            # input_features is a list of length num_nodes
            tr_all_node_logits = model(tr_feature, node_failure_mask)  # [B, N, C]

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

                # 在验证集上也应用相同的故障率
                node_failure_mask = None
                if args.failure_rate > 0:
                    node_failure_mask = torch.rand(val_labels.size(0), num_nodes) < args.failure_rate
                    node_failure_mask = node_failure_mask.to(device)

                val_all_node_logits = model(val_feature, node_failure_mask)  # [B, N, C]

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
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

        # early_stopper(val_acc_mean)
        # if early_stopper.early_stop:
        #     break

    # draw_fig(history, epoch, save_path=f"plot_underwater_dg{args.dg}_fail{args.failure_rate}.png")

    # --- Testing ---
    print("--- Starting Testing ---")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    te_pernode_allbatch_corr = []
    all_te_node_logits = []
    all_te_labels = []

    with torch.no_grad():
        for te_feature, te_labels in te_loader:
            te_labels = te_labels.to(device)

            node_failure_mask = None
            if args.failure_rate > 0:
                node_failure_mask = torch.rand(te_labels.size(0), num_nodes) < args.failure_rate
                node_failure_mask = node_failure_mask.to(device)

            te_all_node_logits = model(te_feature, node_failure_mask)  # [B, N, C]

            te_each_current_correct = eachnode_correct(te_all_node_logits, te_labels)
            te_pernode_allbatch_corr.append(te_each_current_correct)
            all_te_node_logits.append(te_all_node_logits.detach())
            all_te_labels.extend(te_labels.cpu().numpy())

    te_acc_mean, te_acc_max, te_acc_min, te_F1_mean, te_F1_max, te_F1_min, te_per_node_acc, te_per_node_f1 = acc_F1_distributed(
        te_pernode_allbatch_corr, all_te_labels, all_te_node_logits, num_nodes
    )

    print("\n--- Results for this Single Run ---")
    print(f"Mean Accuracy: {te_acc_mean:.2f}%")
    print(f"Mean Macro F1-Score: {te_F1_mean:.2f}%")
    print(f"Best-Node Accuracy: {te_acc_max:.2f}%")
    print(f"Best-Node Macro F1-Score: {te_F1_max:.2f}%")
    print(f"Worst-Node Accuracy: {te_acc_min:.2f}%")
    print(f"Worst-Node Macro F1-Score: {te_F1_min:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Underwater Acoustic Classification Experiments")
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Directory where the .pt files are stored.')
    parser.add_argument('--panns_path', type=str, default='./models/MobileNetV2_mAP=0.383.pth',
                        help='Path to the pretrained PANNs model.')
    parser.add_argument('--dg', type=int, default=32,
                        help='Compressed feature dimension (d_g). e.g., 8, 32')
    parser.add_argument('--weighting', action='store_true',
                        help="Use SE Weighting. If not set, uses no weighting.")
    parser.add_argument('--failure_rate', type=float, default=0.2,
                        help='Node failure rate. e.g., 0, 0.2, 0.4, 0.6')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    args = parser.parse_args()
    run_experiment(args)