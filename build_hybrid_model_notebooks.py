from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
HYBRID_NOTEBOOKS = ROOT / "hybrid_models"
HYBRID_NOTEBOOKS.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(path: Path, cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    with path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


cells = [
    md(
        """
        # 01. Hybrid Risk Scoring

        This notebook combines four complementary wash-trading signals:

        - `tabular_score`: wallet feature model
        - `motif_score`: graph motif and local-structure model
        - `gnn_score`: GraphSAGE message-passing model
        - `anomaly_score`: unsupervised anomaly detector

        The final layer is a calibrated logistic meta-model trained on the validation split.

        Presentation framing:

        > Instead of one classifier, we built a multi-signal fraud intelligence system.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import pandas as pd

        ROOT = Path.cwd().resolve().parent if Path.cwd().name == "hybrid_models" else Path.cwd().resolve()
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))

        from hybrid_models.hybrid_model_utils import (
            plot_evaluation_dashboard,
            plot_hybrid_comparison_bars,
            plot_meta_coefficients,
            plot_signal_correlation,
            run_hybrid_experiment,
            top_hybrid_alerts,
        )
        """
    ),
    code(
        """
        FEATURE_GROUP = "features"
        ADD_GRAPH_STATS = False
        RANDOM_STATE = 42
        TABULAR_MODEL_NAME = "xgboost"
        MOTIF_MODEL_NAME = "xgboost"
        GNN_MODEL_NAME = "graphsage"
        GNN_HIDDEN_DIM = 64
        GNN_DROPOUT = 0.2
        GNN_LEARNING_RATE = 1e-3
        GNN_WEIGHT_DECAY = 1e-4
        GNN_EPOCHS = 100
        GNN_PATIENCE = 15
        THRESHOLD_OBJECTIVE = "f1"
        """
    ),
    code(
        """
        experiment = run_hybrid_experiment(
            feature_group=FEATURE_GROUP,
            add_graph_stats=ADD_GRAPH_STATS,
            random_state=RANDOM_STATE,
            tabular_model_name=TABULAR_MODEL_NAME,
            motif_model_name=MOTIF_MODEL_NAME,
            gnn_model_name=GNN_MODEL_NAME,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            gnn_dropout=GNN_DROPOUT,
            gnn_learning_rate=GNN_LEARNING_RATE,
            gnn_weight_decay=GNN_WEIGHT_DECAY,
            gnn_epochs=GNN_EPOCHS,
            gnn_patience=GNN_PATIENCE,
            threshold_objective=THRESHOLD_OBJECTIVE,
        )

        display(experiment["comparison_df"])
        plot_hybrid_comparison_bars(experiment["comparison_df"])
        """
    ),
    code(
        """
        plot_evaluation_dashboard(
            y_true=experiment["test_signal_df"]["label"].to_numpy(),
            y_prob=experiment["test_signal_df"]["tabular_score"].to_numpy(),
            threshold=experiment["tabular_signal"]["threshold"],
            title_prefix="Hybrid Tabular Signal",
        )
        """
    ),
    code(
        """
        plot_evaluation_dashboard(
            y_true=experiment["test_signal_df"]["label"].to_numpy(),
            y_prob=experiment["test_signal_df"]["motif_score"].to_numpy(),
            threshold=experiment["motif_signal"]["threshold"],
            title_prefix="Hybrid Motif Signal",
        )
        """
    ),
    code(
        """
        plot_evaluation_dashboard(
            y_true=experiment["test_signal_df"]["label"].to_numpy(),
            y_prob=experiment["test_signal_df"]["gnn_score"].to_numpy(),
            threshold=experiment["gnn_signal"]["threshold"],
            title_prefix="Hybrid GNN Signal",
        )
        """
    ),
    code(
        """
        plot_evaluation_dashboard(
            y_true=experiment["test_signal_df"]["label"].to_numpy(),
            y_prob=experiment["test_signal_df"]["anomaly_score"].to_numpy(),
            threshold=experiment["anomaly_signal"]["threshold"],
            title_prefix="Hybrid Anomaly Signal",
        )
        """
    ),
    code(
        """
        display(experiment["meta_result"]["coefficient_df"])
        plot_meta_coefficients(experiment["meta_result"]["coefficient_df"])
        """
    ),
    code(
        """
        plot_signal_correlation(
            experiment["test_signal_df"],
            title="Hybrid Test-Signal Correlation",
        )
        """
    ),
    code(
        """
        plot_evaluation_dashboard(
            y_true=experiment["test_signal_df"]["label"].to_numpy(),
            y_prob=experiment["meta_result"]["test_prob"],
            threshold=experiment["meta_result"]["threshold"],
            title_prefix="Hybrid Meta-Model",
        )
        """
    ),
    code(
        """
        display(top_hybrid_alerts(experiment["test_signal_df"], top_n=25))
        """
    ),
    code(
        """
        motif_columns = [column for column in experiment["motif_df"].columns if column.startswith("motif_")]
        test_alerts = (
            experiment["test_signal_df"][["node_id", "hybrid_score", "label"]]
            .merge(experiment["motif_df"], on="node_id", how="left")
            .sort_values("hybrid_score", ascending=False)
            .head(15)
        )
        display(test_alerts[["node_id", "label", "hybrid_score"] + motif_columns[:8]])
        """
    ),
    md(
        """
        ## Readout

        If the `hybrid_meta` row wins in `PR-AUC`, `F1`, or `MCC`, you can argue that:

        - tabular features capture wallet-level semantics
        - motif features capture suspicious local trading structure
        - the GNN captures message-passing context
        - anomaly detection adds weak-label robustness
        - the meta-model learns how to combine them into a final risk score
        """
    ),
]


write_notebook(HYBRID_NOTEBOOKS / "01_hybrid_risk_scoring.ipynb", cells)
