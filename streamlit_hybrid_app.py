from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from graph_models.graph_model_utils import classification_report_df
from hybrid_models.hybrid_model_utils import run_hybrid_experiment, top_hybrid_alerts
from tabular_models.tabular_model_utils import evaluate_probabilities, get_feature_groups, load_nodes


TABULAR_MODEL_OPTIONS = [
    "xgboost",
    "random_forest",
    "logistic_regression",
    "mlp",
]
GNN_MODEL_OPTIONS = ["graphsage", "gcn", "gat", "ggnn"]
METRIC_OPTIONS = ["PR-AUC", "F1", "Recall", "Precision", "Balanced-Accuracy", "MCC", "ROC-AUC", "Accuracy"]
SIGNAL_LABELS = {
    "tabular_score": "Tabular",
    "motif_score": "Motif",
    "gnn_score": "GNN",
    "anomaly_score": "Anomaly",
    "hybrid_score": "Hybrid",
}
SIGNAL_TO_RESULT_KEY = {
    "Tabular": "tabular_signal",
    "Motif": "motif_signal",
    "GNN": "gnn_signal",
    "Anomaly": "anomaly_signal",
    "Hybrid": "meta_result",
}
SIGNAL_TO_SCORE_COL = {
    "Tabular": "tabular_score",
    "Motif": "motif_score",
    "GNN": "gnn_score",
    "Anomaly": "anomaly_score",
    "Hybrid": "hybrid_score",
}


@st.cache_data(show_spinner=False)
def available_feature_groups() -> list[str]:
    nodes_df = load_nodes()
    feature_groups = get_feature_groups(nodes_df)
    return ["features"] if "features" in feature_groups else sorted(feature_groups)


@st.cache_resource(show_spinner=False)
def run_cached_experiment(
    feature_group: str,
    add_graph_stats: bool,
    tabular_model_name: str,
    motif_model_name: str,
    gnn_model_name: str,
    random_state: int,
    gnn_hidden_dim: int,
    gnn_dropout: float,
    gnn_learning_rate: float,
    gnn_weight_decay: float,
    gnn_epochs: int,
    gnn_patience: int,
) -> dict:
    return run_hybrid_experiment(
        feature_group=feature_group,
        add_graph_stats=add_graph_stats,
        random_state=random_state,
        tabular_model_name=tabular_model_name,
        motif_model_name=motif_model_name,
        gnn_model_name=gnn_model_name,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_dropout=gnn_dropout,
        gnn_learning_rate=gnn_learning_rate,
        gnn_weight_decay=gnn_weight_decay,
        gnn_epochs=gnn_epochs,
        gnn_patience=gnn_patience,
        threshold_objective="f1",
    )


def format_metric_delta(current: float, baseline: float) -> str:
    return f"{current - baseline:+.3f}"


def best_single_signal(comparison_df: pd.DataFrame) -> pd.Series:
    singles = comparison_df.loc[comparison_df["signal"] != "hybrid_meta"].copy()
    singles = singles.sort_values(["PR-AUC", "F1"], ascending=False)
    return singles.iloc[0]


def metric_bar_figure(comparison_df: pd.DataFrame, metric: str) -> Figure:
    plot_df = comparison_df.copy()
    plot_df["display"] = plot_df["signal"].replace({"hybrid_meta": "hybrid"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="display", y=metric, ax=ax, palette="Blues_d")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by signal")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def meta_weight_figure(coefficient_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=coefficient_df, x="signal", y="coefficient", ax=ax, palette="crest")
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient")
    ax.set_title("Meta-model signal weights")
    fig.tight_layout()
    return fig


def correlation_figure(signal_df: pd.DataFrame) -> Figure:
    score_cols = [c for c in ["tabular_score", "motif_score", "gnn_score", "anomaly_score", "hybrid_score"] if c in signal_df]
    renamed = signal_df[score_cols].rename(columns=SIGNAL_LABELS)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(renamed.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Signal correlation")
    fig.tight_layout()
    return fig


def confusion_figure(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, title: str) -> Figure:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title(f"{title} confusion @ {threshold:.2f}")
    fig.tight_layout()
    return fig


def pr_roc_figure(y_true: np.ndarray, y_prob: np.ndarray, title: str) -> Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(recall, precision, color="#1f77b4")
    axes[0].set_title(f"{title} PR curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")

    axes[1].plot(fpr, tpr, color="#2ca02c", label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title(f"{title} ROC curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    fig.tight_layout()
    return fig


def threshold_figure(y_true: np.ndarray, y_prob: np.ndarray, chosen_threshold: float, title: str) -> Figure:
    threshold_grid = np.linspace(0.05, 0.95, 37)
    rows = []
    for threshold in threshold_grid:
        metrics = evaluate_probabilities(y_true, y_prob, float(threshold))
        rows.append(
            {
                "threshold": threshold,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "Specificity": metrics["Specificity"],
            }
        )
    threshold_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for metric in ["Precision", "Recall", "F1", "Specificity"]:
        ax.plot(threshold_df["threshold"], threshold_df[metric], label=metric)
    ax.axvline(chosen_threshold, color="red", linestyle="--", label="Chosen threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{title} metrics vs threshold")
    ax.legend()
    fig.tight_layout()
    return fig


def calibration_figure(y_true: np.ndarray, y_prob: np.ndarray, title: str) -> Figure:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"{title} calibration")
    ax.legend()
    fig.tight_layout()
    return fig


def score_distribution_figure(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.65, label="label=0")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.65, label="label=1")
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} score distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def signal_arrays(experiment: dict, signal_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = experiment["test_signal_df"]["label"].to_numpy(copy=True)
    if signal_name == "Hybrid":
        return (
            y_true,
            experiment["meta_result"]["test_prob"],
            float(experiment["meta_result"]["threshold"]),
        )

    score_col = SIGNAL_TO_SCORE_COL[signal_name]
    result_key = SIGNAL_TO_RESULT_KEY[signal_name]
    return (
        y_true,
        experiment["test_signal_df"][score_col].to_numpy(copy=True),
        float(experiment[result_key]["threshold"]),
    )


def experiment_params(
    feature_group: str,
    add_graph_stats: bool,
    tabular_model_name: str,
    motif_model_name: str,
    gnn_model_name: str,
    random_state: int,
    gnn_hidden_dim: int,
    gnn_dropout: float,
    gnn_learning_rate: float,
    gnn_weight_decay: float,
    gnn_epochs: int,
    gnn_patience: int,
) -> dict:
    return {
        "feature_group": feature_group,
        "add_graph_stats": add_graph_stats,
        "tabular_model_name": tabular_model_name,
        "motif_model_name": motif_model_name,
        "gnn_model_name": gnn_model_name,
        "random_state": int(random_state),
        "gnn_hidden_dim": int(gnn_hidden_dim),
        "gnn_dropout": float(gnn_dropout),
        "gnn_learning_rate": float(gnn_learning_rate),
        "gnn_weight_decay": float(gnn_weight_decay),
        "gnn_epochs": int(gnn_epochs),
        "gnn_patience": int(gnn_patience),
    }


def main() -> None:
    st.set_page_config(page_title="Hybrid Model Comparison", layout="wide")
    st.title("Hybrid Wash-Trading Model Dashboard")
    st.caption("Interactive view over the hybrid risk-scoring experiment, base-signal comparison, and top alerts.")

    with st.sidebar:
        st.header("Experiment Settings")
        feature_group = "features"
        st.caption("Feature group fixed to `features`.")
        add_graph_stats = st.checkbox("Add graph stats to base features", value=False)
        tabular_model_name = st.selectbox("Tabular model", TABULAR_MODEL_OPTIONS, index=0)
        motif_model_name = st.selectbox("Motif model", TABULAR_MODEL_OPTIONS, index=0)
        gnn_model_name = st.selectbox("GNN model", GNN_MODEL_OPTIONS, index=0)
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
        quick_mode = st.checkbox("Quick mode", value=True, help="Uses fewer GNN epochs so the dashboard becomes usable faster.")

        with st.expander("Advanced GNN settings"):
            default_epochs = 20 if quick_mode else 100
            default_patience = 5 if quick_mode else 15
            gnn_hidden_dim = st.slider("Hidden dimension", min_value=16, max_value=256, value=64, step=16)
            gnn_dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
            gnn_learning_rate = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
            )
            gnn_weight_decay = st.number_input(
                "Weight decay",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f",
            )
            gnn_epochs = st.slider("Epochs", min_value=5, max_value=200, value=default_epochs, step=5)
            gnn_patience = st.slider("Patience", min_value=1, max_value=40, value=default_patience, step=1)

        requested_params = experiment_params(
            feature_group=feature_group,
            add_graph_stats=add_graph_stats,
            tabular_model_name=tabular_model_name,
            motif_model_name=motif_model_name,
            gnn_model_name=gnn_model_name,
            random_state=int(random_state),
            gnn_hidden_dim=int(gnn_hidden_dim),
            gnn_dropout=float(gnn_dropout),
            gnn_learning_rate=float(gnn_learning_rate),
            gnn_weight_decay=float(gnn_weight_decay),
            gnn_epochs=int(gnn_epochs),
            gnn_patience=int(gnn_patience),
        )

        run_clicked = st.button("Run experiment", type="primary", use_container_width=True)
        st.info(
            "The hybrid run computes motif features, trains tabular and motif models, trains a GNN, and then fits the meta-model. "
            "Use Quick mode for interactive work."
        )

    if "last_experiment" not in st.session_state:
        st.session_state["last_experiment"] = None
        st.session_state["last_params"] = None

    if run_clicked:
        try:
            with st.spinner("Running hybrid experiment..."):
                st.session_state["last_experiment"] = run_cached_experiment(**requested_params)
                st.session_state["last_params"] = requested_params
        except ModuleNotFoundError as exc:
            st.error(str(exc))
            st.stop()

    experiment = st.session_state["last_experiment"]
    last_params = st.session_state["last_params"]

    if experiment is None:
        st.warning("No experiment has been run yet. Pick settings in the sidebar and click `Run experiment`.")
        st.stop()

    if requested_params != last_params:
        st.warning("Sidebar settings changed. Click `Run experiment` to refresh the dashboard with the new configuration.")

    comparison_df = experiment["comparison_df"].reset_index(drop=True).copy()
    best_hybrid = comparison_df.loc[comparison_df["signal"] == "hybrid_meta"].iloc[0]
    best_single = best_single_signal(comparison_df)

    st.subheader("Hybrid vs Best Single Signal")
    st.caption(
        f"Baseline single model: `{str(best_single['signal']).title()}`. "
        "The metric cards below show the hybrid score and the improvement over that baseline."
    )

    summary_cols = st.columns(6)
    summary_cols[0].metric("Hybrid model", "Hybrid")
    summary_cols[1].metric("Best single model", str(best_single["signal"]).title())
    summary_cols[2].metric(
        "Hybrid PR-AUC",
        f"{best_hybrid['PR-AUC']:.3f}",
        format_metric_delta(best_hybrid["PR-AUC"], best_single["PR-AUC"]),
    )
    summary_cols[3].metric(
        "Hybrid F1",
        f"{best_hybrid['F1']:.3f}",
        format_metric_delta(best_hybrid["F1"], best_single["F1"]),
    )
    summary_cols[4].metric(
        "Hybrid Recall",
        f"{best_hybrid['Recall']:.3f}",
        format_metric_delta(best_hybrid["Recall"], best_single["Recall"]),
    )
    summary_cols[5].metric(
        "Hybrid Balanced Accuracy",
        f"{best_hybrid['Balanced-Accuracy']:.3f}",
        format_metric_delta(best_hybrid["Balanced-Accuracy"], best_single["Balanced-Accuracy"]),
    )

    compare_df = pd.DataFrame(
        [
            {
                "model": "Hybrid",
                "PR-AUC": best_hybrid["PR-AUC"],
                "F1": best_hybrid["F1"],
                "Recall": best_hybrid["Recall"],
                "Balanced-Accuracy": best_hybrid["Balanced-Accuracy"],
            },
            {
                "model": str(best_single["signal"]).title(),
                "PR-AUC": best_single["PR-AUC"],
                "F1": best_single["F1"],
                "Recall": best_single["Recall"],
                "Balanced-Accuracy": best_single["Balanced-Accuracy"],
            },
        ]
    )
    st.dataframe(compare_df.style.format({"PR-AUC": "{:.3f}", "F1": "{:.3f}", "Recall": "{:.3f}", "Balanced-Accuracy": "{:.3f}"}), use_container_width=True, hide_index=True)

    tabs = st.tabs(["Overview", "Model Weights", "Evaluation", "Top Alerts"])

    with tabs[0]:
        left, right = st.columns([1.15, 1.0])
        with left:
            st.markdown("#### Comparison Table")
            display_df = comparison_df.copy()
            numeric_cols = display_df.select_dtypes(include=["number"]).columns
            st.dataframe(display_df.style.format({col: "{:.3f}" for col in numeric_cols if col not in ["TP", "FP", "TN", "FN", "TopK"]}), use_container_width=True)

            metric = st.selectbox("Bar chart metric", METRIC_OPTIONS, index=0)
            st.pyplot(metric_bar_figure(comparison_df, metric), clear_figure=True)

        with right:
            st.markdown("#### Dataset / Split Summary")
            graph_summary_df = pd.DataFrame(
                [
                    {"item": key, "value": value}
                    for key, value in experiment["graph_data"]["graph_summary"].items()
                ]
            )
            st.dataframe(graph_summary_df, use_container_width=True, hide_index=True)
            st.dataframe(experiment["tabular_dataset"]["split_df"], use_container_width=True, hide_index=True)

            st.markdown("#### GNN Training History")
            history_df = experiment["gnn_signal"]["history_df"]
            st.line_chart(history_df.set_index("epoch")[["train_loss", "val_loss"]])
            st.line_chart(history_df.set_index("epoch")[["val_PR-AUC", "val_F1", "val_Recall"]])

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Meta-Model Signal Weights")
            st.pyplot(meta_weight_figure(experiment["meta_result"]["coefficient_df"]), clear_figure=True)
            st.dataframe(experiment["meta_result"]["coefficient_df"].style.format({"coefficient": "{:.4f}"}), use_container_width=True, hide_index=True)

        with right:
            st.markdown("#### Signal Correlation")
            st.pyplot(correlation_figure(experiment["test_signal_df"]), clear_figure=True)
            st.dataframe(
                experiment["test_signal_df"][
                    ["tabular_score", "motif_score", "gnn_score", "anomaly_score", "hybrid_score"]
                ].rename(columns=SIGNAL_LABELS).corr().style.format("{:.3f}"),
                use_container_width=True,
            )

    with tabs[2]:
        selected_signal = st.selectbox("Signal to inspect", ["Hybrid", "Tabular", "Motif", "GNN", "Anomaly"], index=0)
        y_true, y_prob, threshold = signal_arrays(experiment, selected_signal)
        signal_metrics = evaluate_probabilities(y_true, y_prob, threshold)

        metric_cols = st.columns(6)
        metric_cols[0].metric("Threshold", f"{threshold:.3f}")
        metric_cols[1].metric("PR-AUC", f"{signal_metrics['PR-AUC']:.3f}")
        metric_cols[2].metric("ROC-AUC", f"{signal_metrics['ROC-AUC']:.3f}")
        metric_cols[3].metric("F1", f"{signal_metrics['F1']:.3f}")
        metric_cols[4].metric("Precision", f"{signal_metrics['Precision']:.3f}")
        metric_cols[5].metric("Recall", f"{signal_metrics['Recall']:.3f}")

        top_left, top_right = st.columns(2)
        with top_left:
            st.pyplot(pr_roc_figure(y_true, y_prob, selected_signal), clear_figure=True)
            st.pyplot(threshold_figure(y_true, y_prob, threshold, selected_signal), clear_figure=True)
        with top_right:
            st.pyplot(confusion_figure(y_true, y_prob, threshold, selected_signal), clear_figure=True)
            st.pyplot(score_distribution_figure(y_true, y_prob, threshold, selected_signal), clear_figure=True)
            st.pyplot(calibration_figure(y_true, y_prob, selected_signal), clear_figure=True)

        st.markdown("#### Classification Report")
        report_df = classification_report_df(y_true, y_prob, threshold)
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    with tabs[3]:
        st.markdown("#### Highest-Risk Hybrid Alerts")
        top_n = st.slider("Rows to show", min_value=10, max_value=100, value=25, step=5)
        st.dataframe(top_hybrid_alerts(experiment["test_signal_df"], top_n=top_n), use_container_width=True, hide_index=True)

        st.markdown("#### Hybrid Alerts With Motif Features")
        motif_columns = [column for column in experiment["motif_df"].columns if column.startswith("motif_")]
        alert_df = (
            experiment["test_signal_df"][["node_id", "hybrid_score", "label"]]
            .merge(experiment["motif_df"], on="node_id", how="left")
            .sort_values("hybrid_score", ascending=False)
            .head(20)
        )
        st.dataframe(alert_df[["node_id", "label", "hybrid_score"] + motif_columns[:8]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
