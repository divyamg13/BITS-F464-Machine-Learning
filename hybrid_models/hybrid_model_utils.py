from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from graph_models.graph_model_utils import (
    build_graph_dataset,
    evaluate_probabilities,
    find_best_threshold,
    get_model,
    plot_evaluation_dashboard,
    predict_probabilities,
    train_model,
)
from tabular_models.tabular_model_utils import build_model, build_tabular_dataset


ROOT = Path(__file__).resolve().parent.parent
NODES_PATH = ROOT / "wash_trading_gnn_nodes_10000.csv"
EDGES_PATH = ROOT / "wash_trading_gnn_edges_10000.csv"


def _mask_to_index(mask) -> np.ndarray:
    return np.flatnonzero(mask.detach().cpu().numpy())


def _normalize_with_train_bounds(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    lower = float(train_scores.min())
    upper = float(train_scores.max())
    if upper <= lower:
        return np.full_like(scores, 0.5, dtype=float)
    return np.clip((scores - lower) / (upper - lower), 0.0, 1.0)


def _signal_frame(
    split_name: str,
    row_idx: np.ndarray,
    nodes_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    score_col: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_idx": row_idx.astype(int),
            "node_id": nodes_df.iloc[row_idx]["node_id"].to_numpy(copy=True),
            "label": y_true.astype(int),
            score_col: y_prob.astype(float),
            "split": split_name,
        }
    )
    return frame


def compute_motif_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    directed_graph = nx.from_pandas_edgelist(
        edges_df,
        source="src_node_id",
        target="dst_node_id",
        create_using=nx.DiGraph(),
    )
    undirected_graph = directed_graph.to_undirected()

    pagerank = nx.pagerank(directed_graph, alpha=0.85)
    in_degree = dict(directed_graph.in_degree())
    out_degree = dict(directed_graph.out_degree())
    total_degree = dict(directed_graph.degree())
    triangle_count = nx.triangles(undirected_graph)
    clustering = nx.clustering(undirected_graph)

    weak_component_size: dict[int, int] = {}
    for component in nx.weakly_connected_components(directed_graph):
        size = len(component)
        for node_id in component:
            weak_component_size[node_id] = size

    succ_map = edges_df.groupby("src_node_id")["dst_node_id"].agg(lambda values: set(values.tolist())).to_dict()
    pred_map = edges_df.groupby("dst_node_id")["src_node_id"].agg(lambda values: set(values.tolist())).to_dict()
    self_loop_nodes = set(
        edges_df.loc[edges_df["src_node_id"] == edges_df["dst_node_id"], "src_node_id"].astype(int).tolist()
    )

    reciprocal_partner_count: dict[int, int] = {}
    reciprocal_ratio: dict[int, float] = {}
    closed_triads_per_degree: dict[int, float] = {}

    for node_id in nodes_df["node_id"].tolist():
        successors = succ_map.get(node_id, set())
        predecessors = pred_map.get(node_id, set())
        reciprocal_partners = successors & predecessors
        unique_neighbors = successors | predecessors
        reciprocal_partner_count[node_id] = len(reciprocal_partners)
        reciprocal_ratio[node_id] = len(reciprocal_partners) / max(1, len(unique_neighbors))
        closed_triads_per_degree[node_id] = triangle_count.get(node_id, 0) / max(1.0, total_degree.get(node_id, 0))

    motif_df = pd.DataFrame(
        {
            "node_id": nodes_df["node_id"],
            "motif_in_degree": nodes_df["node_id"].map(in_degree).fillna(0.0).astype(float),
            "motif_out_degree": nodes_df["node_id"].map(out_degree).fillna(0.0).astype(float),
            "motif_total_degree": nodes_df["node_id"].map(total_degree).fillna(0.0).astype(float),
            "motif_pagerank": nodes_df["node_id"].map(pagerank).fillna(0.0),
            "motif_component_size": nodes_df["node_id"].map(weak_component_size).fillna(1.0).astype(float),
            "motif_triangle_count": nodes_df["node_id"].map(triangle_count).fillna(0.0).astype(float),
            "motif_clustering": nodes_df["node_id"].map(clustering).fillna(0.0),
            "motif_reciprocal_partner_count": nodes_df["node_id"]
            .map(reciprocal_partner_count)
            .fillna(0.0)
            .astype(float),
            "motif_reciprocal_ratio": nodes_df["node_id"].map(reciprocal_ratio).fillna(0.0),
            "motif_closed_triads_per_degree": nodes_df["node_id"].map(closed_triads_per_degree).fillna(0.0),
            "motif_has_self_loop": nodes_df["node_id"].isin(self_loop_nodes).astype(int),
        }
    )
    return motif_df


def _fit_probability_signal(
    model: Pipeline,
    dataset: dict,
    feature_frame: pd.DataFrame,
    score_col: str,
    threshold_objective: str = "f1",
) -> dict:
    train_idx = dataset["train_idx"]
    val_idx = dataset["val_idx"]
    test_idx = dataset["test_idx"]

    X_train = feature_frame.loc[train_idx]
    X_val = feature_frame.loc[val_idx]
    X_test = feature_frame.loc[test_idx]

    y_train = np.asarray(dataset["y_train"].to_numpy(copy=True) if hasattr(dataset["y_train"], "to_numpy") else dataset["y_train"])
    y_val = dataset["y_val"].to_numpy(copy=True)
    y_test = dataset["y_test"].to_numpy(copy=True)

    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    threshold = find_best_threshold(y_val, val_prob, objective=threshold_objective)
    metrics = evaluate_probabilities(y_test, test_prob, threshold)

    val_frame = _signal_frame("val", np.asarray(val_idx), dataset["nodes_df"], y_val, val_prob, score_col)
    test_frame = _signal_frame("test", np.asarray(test_idx), dataset["nodes_df"], y_test, test_prob, score_col)

    return {
        "model": model,
        "threshold": threshold,
        "metrics": metrics,
        "train_prob": train_prob,
        "val_frame": val_frame,
        "test_frame": test_frame,
    }


def fit_tabular_signal(
    dataset: dict,
    model_name: str = "xgboost",
    random_state: int = 42,
    threshold_objective: str = "f1",
) -> dict:
    model = build_model(model_name, dataset["scale_pos_weight"], random_state=random_state)
    return _fit_probability_signal(
        model=model,
        dataset=dataset,
        feature_frame=dataset["nodes_df"][dataset["feature_cols"]],
        score_col="tabular_score",
        threshold_objective=threshold_objective,
    )


def build_motif_dataset(
    nodes_df: pd.DataFrame,
    motif_df: pd.DataFrame,
    split_idx: dict,
) -> dict:
    feature_cols = [column for column in motif_df.columns if column != "node_id"]
    merged_df = nodes_df[["node_id", "label"]].merge(motif_df, on="node_id", how="left")
    return {
        "nodes_df": merged_df,
        "feature_cols": feature_cols,
        "train_idx": split_idx["train_idx"],
        "val_idx": split_idx["val_idx"],
        "test_idx": split_idx["test_idx"],
        "y_train": merged_df.loc[split_idx["train_idx"], "label"].copy(),
        "y_val": merged_df.loc[split_idx["val_idx"], "label"].copy(),
        "y_test": merged_df.loc[split_idx["test_idx"], "label"].copy(),
        "scale_pos_weight": merged_df.loc[split_idx["train_idx"], "label"].eq(0).sum()
        / merged_df.loc[split_idx["train_idx"], "label"].eq(1).sum(),
    }


def fit_motif_signal(
    motif_dataset: dict,
    model_name: str = "xgboost",
    random_state: int = 42,
    threshold_objective: str = "f1",
) -> dict:
    model = build_model(model_name, motif_dataset["scale_pos_weight"], random_state=random_state)
    return _fit_probability_signal(
        model=model,
        dataset=motif_dataset,
        feature_frame=motif_dataset["nodes_df"][motif_dataset["feature_cols"]],
        score_col="motif_score",
        threshold_objective=threshold_objective,
    )


def fit_gnn_signal(
    data: dict,
    model_name: str = "graphsage",
    hidden_dim: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    threshold_objective: str = "f1",
) -> dict:
    model = get_model(
        model_name=model_name,
        in_dim=data["features"].shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    best_model, history_df = train_model(
        model=model,
        data=data,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        threshold_objective=threshold_objective,
    )

    train_true, train_prob = predict_probabilities(best_model, data, "train_mask")
    val_true, val_prob = predict_probabilities(best_model, data, "val_mask")
    test_true, test_prob = predict_probabilities(best_model, data, "test_mask")

    threshold = find_best_threshold(val_true, val_prob, objective=threshold_objective)
    metrics = evaluate_probabilities(test_true, test_prob, threshold)

    val_idx = _mask_to_index(data["val_mask"])
    test_idx = _mask_to_index(data["test_mask"])

    return {
        "model": best_model,
        "history_df": history_df,
        "threshold": threshold,
        "metrics": metrics,
        "train_prob": train_prob,
        "val_frame": _signal_frame("val", val_idx, data["nodes_df"], val_true, val_prob, "gnn_score"),
        "test_frame": _signal_frame("test", test_idx, data["nodes_df"], test_true, test_prob, "gnn_score"),
    }


def fit_anomaly_signal(
    nodes_df: pd.DataFrame,
    split_idx: dict,
    feature_cols: list[str],
    score_col: str = "anomaly_score",
    random_state: int = 42,
    threshold_objective: str = "f1",
) -> dict:
    feature_frame = nodes_df[feature_cols].copy()
    train_idx = np.asarray(split_idx["train_idx"])
    val_idx = np.asarray(split_idx["val_idx"])
    test_idx = np.asarray(split_idx["test_idx"])

    y_train = nodes_df.loc[train_idx, "label"].to_numpy(copy=True)
    y_val = nodes_df.loc[val_idx, "label"].to_numpy(copy=True)
    y_test = nodes_df.loc[test_idx, "label"].to_numpy(copy=True)

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train = preprocessor.fit_transform(feature_frame.loc[train_idx])
    X_val = preprocessor.transform(feature_frame.loc[val_idx])
    X_test = preprocessor.transform(feature_frame.loc[test_idx])

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    anomaly_model.fit(X_train[y_train == 0])

    train_raw = -anomaly_model.decision_function(X_train)
    val_raw = -anomaly_model.decision_function(X_val)
    test_raw = -anomaly_model.decision_function(X_test)

    val_prob = _normalize_with_train_bounds(train_raw, val_raw)
    test_prob = _normalize_with_train_bounds(train_raw, test_raw)

    threshold = find_best_threshold(y_val, val_prob, objective=threshold_objective)
    metrics = evaluate_probabilities(y_test, test_prob, threshold)

    return {
        "preprocessor": preprocessor,
        "model": anomaly_model,
        "threshold": threshold,
        "metrics": metrics,
        "val_frame": _signal_frame("val", val_idx, nodes_df, y_val, val_prob, score_col),
        "test_frame": _signal_frame("test", test_idx, nodes_df, y_test, test_prob, score_col),
    }


def _merge_signal_frames(signal_frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = signal_frames[0].copy()
    for frame in signal_frames[1:]:
        merged = merged.merge(
            frame.drop(columns=["label", "split"]),
            on=["row_idx", "node_id"],
            how="inner",
        )
    return merged


def fit_meta_model(
    val_signal_df: pd.DataFrame,
    test_signal_df: pd.DataFrame,
    random_state: int = 42,
    threshold_objective: str = "f1",
) -> dict:
    signal_cols = ["tabular_score", "motif_score", "gnn_score", "anomaly_score"]
    meta_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ]
    )
    meta_model.fit(val_signal_df[signal_cols], val_signal_df["label"])

    val_prob = meta_model.predict_proba(val_signal_df[signal_cols])[:, 1]
    test_prob = meta_model.predict_proba(test_signal_df[signal_cols])[:, 1]

    threshold = find_best_threshold(
        val_signal_df["label"].to_numpy(copy=True),
        val_prob,
        objective=threshold_objective,
    )
    metrics = evaluate_probabilities(test_signal_df["label"].to_numpy(copy=True), test_prob, threshold)

    coefficient_df = pd.DataFrame(
        {
            "signal": signal_cols,
            "coefficient": meta_model.named_steps["model"].coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    return {
        "model": meta_model,
        "threshold": threshold,
        "metrics": metrics,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "coefficient_df": coefficient_df,
    }


def run_hybrid_experiment(
    feature_group: str = "features",
    add_graph_stats: bool = False,
    random_state: int = 42,
    tabular_model_name: str = "xgboost",
    motif_model_name: str = "xgboost",
    gnn_model_name: str = "graphsage",
    gnn_hidden_dim: int = 64,
    gnn_dropout: float = 0.2,
    gnn_learning_rate: float = 1e-3,
    gnn_weight_decay: float = 1e-4,
    gnn_epochs: int = 100,
    gnn_patience: int = 15,
    threshold_objective: str = "f1",
) -> dict:
    tabular_dataset = build_tabular_dataset(
        feature_group=feature_group,
        add_graph_stats=add_graph_stats,
        random_state=random_state,
    )
    graph_data = build_graph_dataset(
        feature_group=feature_group,
        add_graph_stats=add_graph_stats,
        random_state=random_state,
    )

    graph_train_idx = _mask_to_index(graph_data["train_mask"])
    graph_val_idx = _mask_to_index(graph_data["val_mask"])
    graph_test_idx = _mask_to_index(graph_data["test_mask"])
    if not np.array_equal(np.sort(tabular_dataset["train_idx"]), graph_train_idx):
        raise ValueError("Tabular and graph train splits are not aligned.")
    if not np.array_equal(np.sort(tabular_dataset["val_idx"]), graph_val_idx):
        raise ValueError("Tabular and graph validation splits are not aligned.")
    if not np.array_equal(np.sort(tabular_dataset["test_idx"]), graph_test_idx):
        raise ValueError("Tabular and graph test splits are not aligned.")

    nodes_df = tabular_dataset["nodes_df"].copy()
    motif_df = compute_motif_features(nodes_df, graph_data["edges_df"])
    split_idx = {
        "train_idx": tabular_dataset["train_idx"],
        "val_idx": tabular_dataset["val_idx"],
        "test_idx": tabular_dataset["test_idx"],
    }
    motif_dataset = build_motif_dataset(nodes_df, motif_df, split_idx)

    tabular_signal = fit_tabular_signal(
        dataset=tabular_dataset,
        model_name=tabular_model_name,
        random_state=random_state,
        threshold_objective=threshold_objective,
    )
    motif_signal = fit_motif_signal(
        motif_dataset=motif_dataset,
        model_name=motif_model_name,
        random_state=random_state,
        threshold_objective=threshold_objective,
    )
    gnn_signal = fit_gnn_signal(
        data=graph_data,
        model_name=gnn_model_name,
        hidden_dim=gnn_hidden_dim,
        dropout=gnn_dropout,
        learning_rate=gnn_learning_rate,
        weight_decay=gnn_weight_decay,
        epochs=gnn_epochs,
        patience=gnn_patience,
        threshold_objective=threshold_objective,
    )

    anomaly_feature_cols = list(dict.fromkeys(tabular_dataset["feature_cols"] + motif_dataset["feature_cols"]))
    anomaly_input_df = nodes_df.merge(motif_df, on="node_id", how="left")
    anomaly_input_df = anomaly_input_df[["node_id", "label"] + anomaly_feature_cols].copy()
    anomaly_signal = fit_anomaly_signal(
        nodes_df=anomaly_input_df,
        split_idx=split_idx,
        feature_cols=anomaly_feature_cols,
        random_state=random_state,
        threshold_objective=threshold_objective,
    )

    val_signal_df = _merge_signal_frames(
        [
            tabular_signal["val_frame"],
            motif_signal["val_frame"],
            gnn_signal["val_frame"],
            anomaly_signal["val_frame"],
        ]
    )
    test_signal_df = _merge_signal_frames(
        [
            tabular_signal["test_frame"],
            motif_signal["test_frame"],
            gnn_signal["test_frame"],
            anomaly_signal["test_frame"],
        ]
    )

    meta_result = fit_meta_model(
        val_signal_df=val_signal_df,
        test_signal_df=test_signal_df,
        random_state=random_state,
        threshold_objective=threshold_objective,
    )

    test_signal_df = test_signal_df.copy()
    test_signal_df["hybrid_score"] = meta_result["test_prob"]
    test_signal_df["hybrid_pred"] = (test_signal_df["hybrid_score"] >= meta_result["threshold"]).astype(int)

    comparison_df = pd.DataFrame(
        [
            {"signal": "tabular", **tabular_signal["metrics"]},
            {"signal": "motif", **motif_signal["metrics"]},
            {"signal": "gnn", **gnn_signal["metrics"]},
            {"signal": "anomaly", **anomaly_signal["metrics"]},
            {"signal": "hybrid_meta", **meta_result["metrics"]},
        ]
    ).sort_values(["PR-AUC", "F1"], ascending=False)

    return {
        "nodes_df": nodes_df,
        "edges_df": graph_data["edges_df"],
        "motif_df": motif_df,
        "tabular_dataset": tabular_dataset,
        "graph_data": graph_data,
        "motif_dataset": motif_dataset,
        "tabular_signal": tabular_signal,
        "motif_signal": motif_signal,
        "gnn_signal": gnn_signal,
        "anomaly_signal": anomaly_signal,
        "meta_result": meta_result,
        "val_signal_df": val_signal_df,
        "test_signal_df": test_signal_df,
        "comparison_df": comparison_df,
    }


def plot_signal_correlation(signal_df: pd.DataFrame, title: str = "Signal Correlation") -> None:
    score_cols = ["tabular_score", "motif_score", "gnn_score", "anomaly_score", "hybrid_score"]
    available_cols = [column for column in score_cols if column in signal_df.columns]
    plt.figure(figsize=(8, 6))
    sns.heatmap(signal_df[available_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_meta_coefficients(coefficient_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=coefficient_df, x="signal", y="coefficient")
    plt.title("Hybrid Meta-Model Signal Weights")
    plt.tight_layout()
    plt.show()


def plot_hybrid_comparison_bars(comparison_df: pd.DataFrame, metrics: list[str] | None = None) -> None:
    metrics = metrics or ["PR-AUC", "Recall", "Precision", "F1", "Balanced-Accuracy", "MCC"]
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        ax = sns.barplot(data=comparison_df, x="signal", y=metric)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()


def top_hybrid_alerts(test_signal_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    columns = [
        "node_id",
        "label",
        "hybrid_score",
        "hybrid_pred",
        "tabular_score",
        "motif_score",
        "gnn_score",
        "anomaly_score",
    ]
    return test_signal_df.sort_values("hybrid_score", ascending=False).head(top_n)[columns]


__all__ = [
    "compute_motif_features",
    "plot_hybrid_comparison_bars",
    "plot_evaluation_dashboard",
    "plot_meta_coefficients",
    "plot_signal_correlation",
    "run_hybrid_experiment",
    "top_hybrid_alerts",
]
