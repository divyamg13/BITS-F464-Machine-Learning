from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Dict

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GatedGraphConv, GraphConv, SAGEConv
import networkx as nx
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
NODES_PATH = ROOT / "wash_trading_gnn_nodes_10000.csv"
EDGES_PATH = ROOT / "wash_trading_gnn_edges_10000.csv"

# Keep only graph statistics that are computable without using node labels.
GRAPH_STAT_COLS = [
    "full_in_degree",
    "full_out_degree",
    "full_total_degree",
    "full_has_self_loop",
    "sub_in_degree",
    "sub_out_degree",
    "sub_total_degree",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_feature_groups(nodes_df: pd.DataFrame) -> Dict[str, list[str]]:
    return {
        "features": [c for c in nodes_df.columns if c.startswith("features_")],
        "normalized_log_features": [c for c in nodes_df.columns if c.startswith("normalized_log_features_")],
        "twitter_semantic_features": [c for c in nodes_df.columns if c.startswith("twitter_semantic_features_")],
        "twitter_semantic_features_normalized": [
            c for c in nodes_df.columns if c.startswith("twitter_semantic_features_normalized_")
        ],
        "twitter_deepwalk_features": [c for c in nodes_df.columns if c.startswith("twitter_deepwalk_features_")],
        "twitter_deepwalk_normalized_features": [
            c for c in nodes_df.columns if c.startswith("twitter_deepwalk_normalized_features_")
        ],
        "twitter_combined_features": [c for c in nodes_df.columns if c.startswith("twitter_combined_features_")],
        "eth_twitter_combined_features": [c for c in nodes_df.columns if c.startswith("eth_twitter_combined_features_")],
    }


def load_frames(nodes_path: Path | None = None, edges_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path or NODES_PATH)
    edges_df = pd.read_csv(edges_path or EDGES_PATH)
    return nodes_df, edges_df


def build_graph_dataset(
    feature_group: str = "features",
    add_graph_stats: bool = True,
    random_state: int = 42,
    device: str | torch.device | None = None,
) -> dict:
    nodes_df, edges_df = load_frames()
    feature_groups = get_feature_groups(nodes_df)
    if feature_group not in feature_groups:
        raise ValueError(f"Unknown feature_group={feature_group}. Available: {sorted(feature_groups)}")

    feature_cols = list(feature_groups[feature_group])
    if add_graph_stats:
        feature_cols += GRAPH_STAT_COLS

    node_ids = nodes_df["node_id"].tolist()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    src = edges_df["src_node_id"].map(node_to_idx).to_numpy(copy=True)
    dst = edges_df["dst_node_id"].map(node_to_idx).to_numpy(copy=True)

    graph = dgl.graph((src, dst), num_nodes=len(nodes_df))
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    x = torch.tensor(nodes_df[feature_cols].fillna(0.0).to_numpy(copy=True), dtype=torch.float32)
    y = torch.tensor(nodes_df["label"].to_numpy(copy=True), dtype=torch.long)

    train_idx, temp_idx = train_test_split(
        np.arange(len(nodes_df)),
        test_size=0.30,
        random_state=random_state,
        stratify=nodes_df["label"],
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=nodes_df.iloc[temp_idx]["label"],
    )

    train_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    val_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    test_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    train_mean = x[train_mask].mean(dim=0, keepdim=True)
    train_std = x[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - train_mean) / train_std

    class_counts = torch.bincount(y[train_mask], minlength=2)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts.float().clamp_min(1.0))

    split_rows = []
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split_rows.append(
            {
                "split": split_name,
                "rows": int(mask.sum().item()),
                "positives": int(y[mask].sum().item()),
                "positive_ratio": float(y[mask].float().mean().item()),
            }
        )
    split_df = pd.DataFrame(split_rows)

    graph_summary = {
        "num_nodes": len(nodes_df),
        "num_edges": graph.num_edges(),
        "num_features": len(feature_cols),
        "positive_nodes": int(nodes_df["label"].sum()),
        "negative_nodes": int((1 - nodes_df["label"]).sum()),
        "has_self_loops_after_build": True,
    }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    return {
        "graph": graph.to(device),
        "features": x.to(device),
        "labels": y.to(device),
        "train_mask": train_mask.to(device),
        "val_mask": val_mask.to(device),
        "test_mask": test_mask.to(device),
        "class_weights": class_weights.to(device),
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "feature_cols": feature_cols,
        "feature_group": feature_group,
        "add_graph_stats": add_graph_stats,
        "split_df": split_df,
        "graph_summary": graph_summary,
        "device": device,
    }


class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.classifier(h)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.classifier(h)


class GATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2, num_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.gat1(g, features).flatten(1)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(g, h).squeeze(1)
        h = F.elu(h)
        h = self.dropout(h)
        return self.classifier(h)


class GGNNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_steps: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, hidden_dim, n_steps=n_steps, n_etypes=1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, features):
        h = self.input_proj(features)
        etypes = torch.zeros(g.num_edges(), dtype=torch.long, device=features.device)
        h = self.ggnn(g, h, etypes)
        h = F.relu(h)
        h = self.dropout(h)
        return self.classifier(h)


def get_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int = 64,
    out_dim: int = 2,
    dropout: float = 0.2,
    num_heads: int = 4,
    n_steps: int = 3,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCNModel(in_dim, hidden_dim, out_dim, dropout)
    if model_name == "graphsage":
        return GraphSAGEModel(in_dim, hidden_dim, out_dim, dropout)
    if model_name == "gat":
        return GATModel(in_dim, hidden_dim, out_dim, dropout, num_heads)
    if model_name == "ggnn":
        return GGNNModel(in_dim, hidden_dim, out_dim, n_steps, dropout)
    raise ValueError(f"Unsupported model_name={model_name}")


def probability_to_topk_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    positive_count = int(y_true.sum())
    top_k = max(1, positive_count)
    ranked_idx = np.argsort(-y_prob)[:top_k]
    selected_true = y_true[ranked_idx]
    return {
        "TopK": top_k,
        "Precision@K": float(selected_true.mean()),
        "Recall@K": float(selected_true.sum() / max(1, positive_count)),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1") -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold objective={objective}")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    npv = tn / max(1, tn + fn)
    topk_metrics = probability_to_topk_metrics(y_true, y_prob)
    return {
        "Threshold": threshold,
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "NPV": npv,
        "Balanced-Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen-Kappa": cohen_kappa_score(y_true, y_pred),
        "Brier-Score": brier_score_loss(y_true, y_prob),
        "Log-Loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        **topk_metrics,
    }


def classification_report_df(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def train_model(
    model: nn.Module,
    data: dict,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    threshold_objective: str = "f1",
) -> tuple[nn.Module, pd.DataFrame]:
    model = model.to(data["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=data["class_weights"])

    best_val_pr_auc = -1.0
    best_state = None
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(data["graph"], data["features"])
        train_loss = criterion(logits[data["train_mask"]], data["labels"][data["train_mask"]])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data["graph"], data["features"])
            train_prob = torch.softmax(logits[data["train_mask"]], dim=1)[:, 1].detach().cpu().numpy()
            val_prob = torch.softmax(logits[data["val_mask"]], dim=1)[:, 1].detach().cpu().numpy()
            train_true = data["labels"][data["train_mask"]].detach().cpu().numpy()
            val_true = data["labels"][data["val_mask"]].detach().cpu().numpy()
            val_loss = criterion(logits[data["val_mask"]], data["labels"][data["val_mask"]]).item()

        val_threshold = find_best_threshold(val_true, val_prob, objective=threshold_objective)
        train_metrics = evaluate_probabilities(train_true, train_prob, val_threshold)
        val_metrics = evaluate_probabilities(val_true, val_prob, val_threshold)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss),
                "val_threshold": val_threshold,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_metrics["PR-AUC"] > best_val_pr_auc:
            best_val_pr_auc = float(val_metrics["PR-AUC"])
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left == 0:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid best model state.")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_probabilities(model: nn.Module, data: dict, mask_name: str) -> tuple[np.ndarray, np.ndarray]:
    mask = data[mask_name]
    model.eval()
    with torch.no_grad():
        logits = model(data["graph"], data["features"])
        y_prob = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
        y_true = data["labels"][mask].detach().cpu().numpy()
    return y_true, y_prob


def plot_training_history(history_df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["val_PR-AUC"], label="val PR-AUC")
    plt.plot(history_df["epoch"], history_df["val_ROC-AUC"], label="val ROC-AUC")
    plt.title("Ranking Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["val_F1"], label="val F1")
    plt.plot(history_df["epoch"], history_df["val_Precision"], label="val Precision")
    plt.plot(history_df["epoch"], history_df["val_Recall"], label="val Recall")
    plt.title("Thresholded Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history_df["epoch"], history_df["val_threshold"], label="best val threshold")
    plt.title("Threshold Sweep Result")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_evaluation_dashboard(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, title_prefix: str = "") -> None:
    sns.set_theme(style="whitegrid")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    threshold_grid = np.linspace(0.05, 0.95, 37)
    threshold_rows = []
    for grid_threshold in threshold_grid:
        metrics = evaluate_probabilities(y_true, y_prob, grid_threshold)
        threshold_rows.append(
            {
                "threshold": grid_threshold,
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1": metrics["F1"],
                "specificity": metrics["Specificity"],
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{title_prefix} ROC Curve".strip())
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.4f}")
    plt.title(f"{title_prefix} Precision-Recall Curve".strip())
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(f"{title_prefix} Confusion Matrix @ {threshold:.2f}".strip())
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.65, label="label=0")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.65, label="label=1")
    plt.axvline(threshold, color="red", linestyle="--", label="threshold")
    plt.title(f"{title_prefix} Score Distribution".strip())
    plt.xlabel("Predicted probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    plt.plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity")
    plt.axvline(threshold, color="red", linestyle="--", label="chosen threshold")
    plt.title(f"{title_prefix} Metrics vs Threshold".strip())
    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    plt.title(f"{title_prefix} Calibration Curve".strip())
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison_bars(results_df: pd.DataFrame, metrics: list[str] | None = None) -> None:
    metrics = metrics or ["PR-AUC", "Recall", "Precision", "F1", "Balanced-Accuracy", "MCC"]
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        ax = sns.barplot(data=results_df, x="model", y=metric)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()


def plot_component_distribution(edges_df: pd.DataFrame) -> None:
    nx_graph = nx.from_pandas_edgelist(
        edges_df,
        source="src_node_id",
        target="dst_node_id",
        create_using=nx.DiGraph(),
    )
    component_sizes = sorted((len(c) for c in nx.weakly_connected_components(nx_graph)), reverse=True)
    plt.figure(figsize=(8, 4))
    plt.hist(component_sizes, bins=30)
    plt.title("Weakly Connected Component Size Distribution")
    plt.xlabel("Component size")
    plt.ylabel("Frequency")
    plt.show()
