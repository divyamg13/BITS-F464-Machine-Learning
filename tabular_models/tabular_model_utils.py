from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
NODES_PATH = ROOT / "wash_trading_gnn_nodes_10000.csv"

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


def load_nodes(nodes_path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(nodes_path or NODES_PATH)


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


def build_tabular_dataset(
    feature_group: str = "eth_twitter_combined_features",
    add_graph_stats: bool = False,
    random_state: int = 42,
) -> dict:
    nodes_df = load_nodes()
    feature_groups = get_feature_groups(nodes_df)
    if feature_group not in feature_groups:
        raise ValueError(f"Unknown feature_group={feature_group}. Available: {sorted(feature_groups)}")

    feature_cols = list(feature_groups[feature_group])
    if add_graph_stats:
        feature_cols += GRAPH_STAT_COLS

    X = nodes_df[feature_cols].copy()
    y = nodes_df["label"].copy()

    train_idx, temp_idx = train_test_split(
        nodes_df.index,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=y.loc[temp_idx],
    )

    splits = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_train": X.loc[train_idx].copy(),
        "X_val": X.loc[val_idx].copy(),
        "X_test": X.loc[test_idx].copy(),
        "y_train": y.loc[train_idx].copy(),
        "y_val": y.loc[val_idx].copy(),
        "y_test": y.loc[test_idx].copy(),
    }

    split_rows = []
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_part = nodes_df.loc[idx]
        split_rows.append(
            {
                "split": split_name,
                "rows": len(split_part),
                "positives": int(split_part["label"].sum()),
                "positive_ratio": float(split_part["label"].mean()),
            }
        )

    return {
        "nodes_df": nodes_df,
        "feature_cols": feature_cols,
        "feature_group": feature_group,
        "add_graph_stats": add_graph_stats,
        "scale_pos_weight": splits["y_train"].eq(0).sum() / splits["y_train"].eq(1).sum(),
        "split_df": pd.DataFrame(split_rows),
        **splits,
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


def build_model(model_name: str, scale_pos_weight: float, random_state: int = 42) -> Pipeline:
    model_name = model_name.lower()
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "mlp":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=250,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model_name={model_name}")


def fit_and_evaluate_model(
    model_name: str,
    dataset: dict,
    random_state: int = 42,
    threshold_objective: str = "f1",
) -> dict:
    model = build_model(model_name, dataset["scale_pos_weight"], random_state=random_state)
    y_train = (
        dataset["y_train"].to_numpy(copy=True)
        if hasattr(dataset["y_train"], "to_numpy")
        else np.asarray(dataset["y_train"])
    )
    model.fit(dataset["X_train"], y_train)

    val_prob = model.predict_proba(dataset["X_val"])[:, 1]
    threshold = find_best_threshold(dataset["y_val"].to_numpy(), val_prob, objective=threshold_objective)
    test_prob = model.predict_proba(dataset["X_test"])[:, 1]
    metrics = evaluate_probabilities(dataset["y_test"].to_numpy(), test_prob, threshold)

    return {
        "model_name": model_name,
        "model": model,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "threshold": threshold,
        "metrics": metrics,
    }


def plot_evaluation_dashboard(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, title_prefix: str = "") -> None:
    sns.set_theme(style="whitegrid")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title(f"{title_prefix} ROC Curve".strip())
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.4f}")
    axes[1].set_title(f"{title_prefix} Precision-Recall Curve".strip())
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[2],
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    axes[2].set_title(f"{title_prefix} Confusion Matrix @ {threshold:.2f}".strip())

    axes[3].hist(y_prob[y_true == 0], bins=30, alpha=0.65, label="label=0")
    axes[3].hist(y_prob[y_true == 1], bins=30, alpha=0.65, label="label=1")
    axes[3].axvline(threshold, color="red", linestyle="--", label="threshold")
    axes[3].set_title(f"{title_prefix} Score Distribution".strip())
    axes[3].set_xlabel("Predicted probability")
    axes[3].legend()

    axes[4].plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    axes[4].plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    axes[4].plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    axes[4].plot(threshold_df["threshold"], threshold_df["specificity"], label="Specificity")
    axes[4].axvline(threshold, color="red", linestyle="--", label="chosen threshold")
    axes[4].set_title(f"{title_prefix} Metrics vs Threshold".strip())
    axes[4].set_xlabel("Threshold")
    axes[4].legend()

    axes[5].plot(prob_pred, prob_true, marker="o", label="model")
    axes[5].plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    axes[5].set_title(f"{title_prefix} Calibration Curve".strip())
    axes[5].set_xlabel("Mean predicted probability")
    axes[5].set_ylabel("Observed positive rate")
    axes[5].legend()

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: Pipeline, feature_cols: list[str], title: str, top_n: int = 20) -> None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        raw_values = estimator.coef_[0]
        importance = np.abs(raw_values)
        signed = raw_values
    elif hasattr(estimator, "feature_importances_"):
        raw_values = estimator.feature_importances_
        importance = raw_values
        signed = raw_values
    else:
        print("This model does not expose direct feature importances.")
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importance,
            "signed_value": signed,
        }
    ).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(5, top_n * 0.35)))
    sns.barplot(data=importance_df, y="feature", x="importance", orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    display(importance_df)


def plot_mlp_learning_curve(model: Pipeline) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "loss_curve_"):
        print("This model does not expose a loss curve.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(estimator.loss_curve_)
    axes[0].set_title("MLP Loss Curve")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")

    if hasattr(estimator, "validation_scores_") and estimator.validation_scores_ is not None:
        axes[1].plot(estimator.validation_scores_)
        axes[1].set_title("MLP Validation Score")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Validation Score")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_comparison_bars(results_df: pd.DataFrame, metrics: list[str] | None = None) -> None:
    metrics = metrics or ["PR-AUC", "Recall", "Precision", "F1", "Balanced-Accuracy", "MCC"]
    num_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(num_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, metric in zip(axes, metrics):
        sns.barplot(data=results_df, x="model", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[num_metrics:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
