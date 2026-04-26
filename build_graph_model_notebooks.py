from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
GRAPH_NOTEBOOKS = ROOT / "graph_models"
GRAPH_NOTEBOOKS.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


overview_cells = [
    md(
        """
        # 00. Graph Data Overview

        This folder is reserved for graph-only experiments.

        Current scope:

        - `GCN`
        - `GraphSAGE`
        - `GAT`
        - `GGNN`

        Each model notebook uses the same `10,000`-node / `274,561`-edge sampled graph and the same split logic, so the comparisons stay fair.
        By default, these notebooks run in a stricter setup with graph-stat columns disabled.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import networkx as nx
        import pandas as pd
        import seaborn as sns

        sys.path.append(str(Path.cwd()))

        from graph_model_utils import GRAPH_STAT_COLS, get_feature_groups, load_frames

        sns.set_theme(style="whitegrid")
        nodes_df, edges_df = load_frames()
        """
    ),
    code(
        """
        print("nodes_df.shape =", nodes_df.shape)
        print("edges_df.shape =", edges_df.shape)
        display(nodes_df.head())
        display(edges_df.head())
        """
    ),
    code(
        """
        feature_groups = get_feature_groups(nodes_df)
        pd.Series({name: len(cols) for name, cols in feature_groups.items()} | {"graph_stats": len(GRAPH_STAT_COLS)}).sort_values(ascending=False)
        """
    ),
    code(
        """
        label_summary = pd.DataFrame(
            {
                "count": nodes_df["label"].value_counts().sort_index(),
            }
        )
        label_summary["ratio"] = label_summary["count"] / label_summary["count"].sum()
        display(label_summary)

        plt.figure(figsize=(5, 4))
        sns.countplot(data=nodes_df, x="label")
        plt.title("Graph Sample Label Distribution")
        plt.show()
        """
    ),
    code(
        """
        nx_graph = nx.from_pandas_edgelist(
            edges_df,
            source="src_node_id",
            target="dst_node_id",
            create_using=nx.DiGraph(),
        )

        connected_nodes = set(edges_df["src_node_id"]).union(edges_df["dst_node_id"])
        summary = {
            "num_nodes_csv": len(nodes_df),
            "num_edges_csv": len(edges_df),
            "graph_nodes": nx_graph.number_of_nodes(),
            "graph_edges": nx_graph.number_of_edges(),
            "isolated_nodes": int((~nodes_df["node_id"].isin(connected_nodes)).sum()),
            "density": nx.density(nx_graph),
            "self_loops": nx.number_of_selfloops(nx_graph),
            "weak_components": nx.number_weakly_connected_components(nx_graph),
        }
        pd.Series(summary)
        """
    ),
    code(
        """
        component_sizes = sorted((len(c) for c in nx.weakly_connected_components(nx_graph)), reverse=True)
        component_df = pd.DataFrame({"component_size": component_sizes})
        display(component_df.head(10))

        plt.figure(figsize=(7, 4))
        sns.histplot(component_df["component_size"], bins=30)
        plt.title("Weakly Connected Component Size Distribution")
        plt.show()
        """
    ),
    code(
        """
        degree_cols = [
            "full_in_degree",
            "full_out_degree",
            "full_total_degree",
            "sub_in_degree",
            "sub_out_degree",
            "sub_total_degree",
        ]

        display(nodes_df[degree_cols + ["label"]].describe().T)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=nodes_df[["sub_total_degree", "label"]], x="label", y="sub_total_degree")
        plt.yscale("log")
        plt.title("Subgraph Degree By Label")
        plt.show()
        """
    ),
    md(
        """
        ## Next

        Run the model notebooks in this folder:

        - `01_gcn.ipynb`
        - `02_graphsage.ipynb`
        - `03_gat.ipynb`
        - `04_ggnn.ipynb`
        - `05_model_comparison.ipynb`
        """
    ),
]


def model_notebook_cells(model_name: str, title: str, hidden_dim: int, dropout: float, lr: float, epochs: int, patience: int):
    upper = model_name.upper() if model_name != "graphsage" else "GraphSAGE"
    return [
        md(
            f"""
            # {title}

            This notebook trains a `{upper}` node-classification model on the sampled wash-trading graph.

            Included by default:

            - stratified train/val/test split
            - train-time early stopping
            - threshold tuning on validation set
            - rich test metrics
            - loss curves, ROC curve, PR curve, confusion matrix, score histogram, calibration curve

            Default restriction:

            - `ADD_GRAPH_STATS = False`
            - node features are used without explicit graph-stat columns
            """
        ),
        code(
            """
            from pathlib import Path
            import sys

            import pandas as pd

            sys.path.append(str(Path.cwd()))

            from graph_model_utils import (
                build_graph_dataset,
                classification_report_df,
                evaluate_probabilities,
                find_best_threshold,
                get_model,
                plot_evaluation_dashboard,
                plot_training_history,
                predict_probabilities,
                set_seed,
                train_model,
            )
            """
        ),
        code(
            f"""
            MODEL_NAME = "{model_name}"
            FEATURE_GROUP = "features"
            ADD_GRAPH_STATS = False
            RANDOM_STATE = 42
            HIDDEN_DIM = {hidden_dim}
            DROPOUT = {dropout}
            LEARNING_RATE = {lr}
            WEIGHT_DECAY = 1e-4
            EPOCHS = {epochs}
            PATIENCE = {patience}
            THRESHOLD_OBJECTIVE = "f1"

            set_seed(RANDOM_STATE)
            """
        ),
        code(
            """
            data = build_graph_dataset(
                feature_group=FEATURE_GROUP,
                add_graph_stats=ADD_GRAPH_STATS,
                random_state=RANDOM_STATE,
            )

            display(pd.DataFrame([data["graph_summary"]]))
            display(data["split_df"])
            print("Device:", data["device"])
            print("Number of input features:", len(data["feature_cols"]))
            """
        ),
        code(
            """
            model = get_model(
                model_name=MODEL_NAME,
                in_dim=data["features"].shape[1],
                hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT,
            )
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(model)
            print("Trainable parameters:", parameter_count)
            """
        ),
        code(
            """
            best_model, history_df = train_model(
                model=model,
                data=data,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                epochs=EPOCHS,
                patience=PATIENCE,
                threshold_objective=THRESHOLD_OBJECTIVE,
            )

            display(history_df.tail())
            plot_training_history(history_df)
            """
        ),
        code(
            """
            val_true, val_prob = predict_probabilities(best_model, data, "val_mask")
            best_threshold = find_best_threshold(val_true, val_prob, objective=THRESHOLD_OBJECTIVE)

            test_true, test_prob = predict_probabilities(best_model, data, "test_mask")
            test_metrics = evaluate_probabilities(test_true, test_prob, best_threshold)

            metrics_df = pd.DataFrame([test_metrics]).T.rename(columns={0: "value"})
            display(metrics_df)
            """
        ),
        code(
            """
            report_df = classification_report_df(test_true, test_prob, best_threshold)
            display(report_df)
            """
        ),
        code(
            """
            plot_evaluation_dashboard(
                y_true=test_true,
                y_prob=test_prob,
                threshold=best_threshold,
                title_prefix=MODEL_NAME.upper(),
            )
            """
        ),
        code(
            """
            test_node_rows = data["nodes_df"].loc[data["test_mask"].detach().cpu().numpy()].copy()
            test_node_rows["predicted_probability"] = test_prob
            test_node_rows["predicted_label"] = (test_prob >= best_threshold).astype(int)

            suspicious_wallets = test_node_rows.sort_values("predicted_probability", ascending=False).head(25)
            display(
                suspicious_wallets[
                    [
                        "node_id",
                        "label",
                        "predicted_probability",
                        "predicted_label",
                        "full_total_degree",
                        "sub_total_degree",
                    ]
                ]
            )
            """
        ),
        md(
            """
            ## Notes

            Keep the reported `PR-AUC`, `Recall`, `Precision`, `F1`, `Balanced-Accuracy`, and `MCC` together when comparing models.
            The threshold comes from validation data and is reused on the test split.
            """
        ),
    ]


write_notebook(GRAPH_NOTEBOOKS / "00_graph_data_overview.ipynb", overview_cells)
write_notebook(
    GRAPH_NOTEBOOKS / "01_gcn.ipynb",
    model_notebook_cells("gcn", "01. GCN", hidden_dim=64, dropout=0.2, lr=1e-3, epochs=100, patience=15),
)
write_notebook(
    GRAPH_NOTEBOOKS / "02_graphsage.ipynb",
    model_notebook_cells("graphsage", "02. GraphSAGE", hidden_dim=64, dropout=0.2, lr=1e-3, epochs=100, patience=15),
)
write_notebook(
    GRAPH_NOTEBOOKS / "03_gat.ipynb",
    model_notebook_cells("gat", "03. GAT", hidden_dim=32, dropout=0.3, lr=7.5e-4, epochs=120, patience=18),
)
write_notebook(
    GRAPH_NOTEBOOKS / "04_ggnn.ipynb",
    model_notebook_cells("ggnn", "04. GGNN", hidden_dim=64, dropout=0.2, lr=1e-3, epochs=100, patience=15),
)

comparison_cells = [
    md(
        """
        # 05. Graph Model Comparison

        This notebook trains and compares all graph baselines under the same split and feature setup:

        - `GCN`
        - `GraphSAGE`
        - `GAT`
        - `GGNN`

        Each model uses its own default hyperparameter preset, but all models share the same dataset split, evaluation protocol, threshold tuning, and reporting metrics.
        The default comparison is stricter by keeping `ADD_GRAPH_STATS = False`.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys
        import time

        import pandas as pd

        sys.path.append(str(Path.cwd()))

        from graph_model_utils import (
            build_graph_dataset,
            find_best_threshold,
            get_model,
            plot_comparison_bars,
            plot_evaluation_dashboard,
            predict_probabilities,
            set_seed,
            train_model,
            evaluate_probabilities,
        )
        """
    ),
    code(
        """
        FEATURE_GROUP = "features"
        ADD_GRAPH_STATS = False
        RANDOM_STATE = 42
        THRESHOLD_OBJECTIVE = "f1"

        MODEL_CONFIGS = {
            "gcn": {"hidden_dim": 64, "dropout": 0.2, "learning_rate": 1e-3, "epochs": 100, "patience": 15},
            "graphsage": {"hidden_dim": 64, "dropout": 0.2, "learning_rate": 1e-3, "epochs": 100, "patience": 15},
            "gat": {"hidden_dim": 32, "dropout": 0.3, "learning_rate": 7.5e-4, "epochs": 120, "patience": 18},
            "ggnn": {"hidden_dim": 64, "dropout": 0.2, "learning_rate": 1e-3, "epochs": 100, "patience": 15},
        }

        set_seed(RANDOM_STATE)
        """
    ),
    code(
        """
        data = build_graph_dataset(
            feature_group=FEATURE_GROUP,
            add_graph_stats=ADD_GRAPH_STATS,
            random_state=RANDOM_STATE,
        )

        display(pd.DataFrame([data["graph_summary"]]))
        display(data["split_df"])
        print("Device:", data["device"])
        print("Number of input features:", len(data["feature_cols"]))
        """
    ),
    code(
        """
        results = []
        fitted = {}
        histories = {}

        for model_name, cfg in MODEL_CONFIGS.items():
            set_seed(RANDOM_STATE)
            model = get_model(
                model_name=model_name,
                in_dim=data["features"].shape[1],
                hidden_dim=cfg["hidden_dim"],
                dropout=cfg["dropout"],
            )

            start = time.perf_counter()
            best_model, history_df = train_model(
                model=model,
                data=data,
                learning_rate=cfg["learning_rate"],
                weight_decay=1e-4,
                epochs=cfg["epochs"],
                patience=cfg["patience"],
                threshold_objective=THRESHOLD_OBJECTIVE,
            )
            elapsed = time.perf_counter() - start

            val_true, val_prob = predict_probabilities(best_model, data, "val_mask")
            threshold = find_best_threshold(val_true, val_prob, objective=THRESHOLD_OBJECTIVE)
            test_true, test_prob = predict_probabilities(best_model, data, "test_mask")
            metrics = evaluate_probabilities(test_true, test_prob, threshold)

            results.append(
                {
                    "model": model_name,
                    "epochs_ran": int(history_df["epoch"].max()),
                    "train_seconds": elapsed,
                    **metrics,
                }
            )
            fitted[model_name] = {
                "model": best_model,
                "threshold": threshold,
                "test_true": test_true,
                "test_prob": test_prob,
                "history": history_df,
            }
            histories[model_name] = history_df
        """
    ),
    code(
        """
        results_df = pd.DataFrame(results).sort_values(["PR-AUC", "F1"], ascending=False)
        display(results_df)
        """
    ),
    code(
        """
        plot_comparison_bars(results_df)
        """
    ),
    code(
        """
        best_model_name = results_df.iloc[0]["model"]
        print("Best graph model:", best_model_name)

        best_result = fitted[best_model_name]
        plot_evaluation_dashboard(
            y_true=best_result["test_true"],
            y_prob=best_result["test_prob"],
            threshold=best_result["threshold"],
            title_prefix=best_model_name.upper(),
        )
        """
    ),
    code(
        """
        summary_cols = [
            "model",
            "PR-AUC",
            "ROC-AUC",
            "F1",
            "Precision",
            "Recall",
            "Specificity",
            "Balanced-Accuracy",
            "MCC",
            "Brier-Score",
            "Accuracy",
            "Precision@K",
            "Recall@K",
            "epochs_ran",
            "train_seconds",
        ]
        results_df[summary_cols]
        """
    ),
    md(
        """
        ## Notes

        This notebook is the main leaderboard for graph models.
        It defaults to a stricter feature-only setup by keeping `ADD_GRAPH_STATS = False`.
        If you later tune hyperparameters, keep the same split and threshold-selection protocol so the comparison remains valid.
        """
    ),
]

write_notebook(GRAPH_NOTEBOOKS / "05_model_comparison.ipynb", comparison_cells)

print(f"Wrote graph model notebooks to {GRAPH_NOTEBOOKS}")
