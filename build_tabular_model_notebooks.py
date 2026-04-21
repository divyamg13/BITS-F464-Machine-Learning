from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
TABULAR_NOTEBOOKS = ROOT / "tabular_models"
TABULAR_NOTEBOOKS.mkdir(exist_ok=True)


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
        # 00. Tabular Data Overview

        This folder is reserved for non-graph baselines:

        - Logistic Regression
        - Random Forest
        - MLP
        - XGBoost

        All notebooks use the same `10,000`-row wallet table and the same stratified split logic so the comparisons stay fair.
        By default, these notebooks run in a stricter feature-only mode and do not include graph-stat columns.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sys.path.append(str(Path.cwd()))

        from tabular_model_utils import GRAPH_STAT_COLS, get_feature_groups, load_nodes

        sns.set_theme(style="whitegrid")
        nodes_df = load_nodes()
        """
    ),
    code(
        """
        print("nodes_df.shape =", nodes_df.shape)
        display(nodes_df.head())
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
        label_summary = pd.DataFrame({"count": nodes_df["label"].value_counts().sort_index()})
        label_summary["ratio"] = label_summary["count"] / label_summary["count"].sum()
        display(label_summary)

        plt.figure(figsize=(5, 4))
        sns.countplot(data=nodes_df, x="label")
        plt.title("Tabular Sample Label Distribution")
        plt.show()
        """
    ),
    code(
        """
        nan_counts = nodes_df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        print("Columns with NaNs:", len(nan_counts))
        display(nan_counts.head(20))
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
        sns.boxplot(data=nodes_df[["label", "sub_total_degree"]], x="label", y="sub_total_degree")
        plt.yscale("log")
        plt.title("Subgraph Degree By Label")
        plt.show()
        """
    ),
    md(
        """
        ## Next

        Run the model notebooks in this folder:

        - `01_logistic_regression.ipynb`
        - `02_random_forest.ipynb`
        - `03_mlp.ipynb`
        - `04_xgboost.ipynb`
        - `05_model_comparison.ipynb`
        """
    ),
]


def model_notebook_cells(model_name: str, title: str, importance_supported: bool, include_learning_curve: bool = False):
    pretty_name = title.split(". ", 1)[1]
    return [
        md(
            f"""
            # {title}

            This notebook trains a `{pretty_name}` baseline on the wallet-level table.

            Included by default:

            - stratified train/val/test split
            - validation-based threshold tuning
            - rich test metrics
            - ROC curve, PR curve, confusion matrix, score histogram, threshold sweeps, calibration curve

            Default restriction:

            - `ADD_GRAPH_STATS = False`
            - only wallet feature columns are used
            """
        ),
        code(
            """
            from pathlib import Path
            import sys

            import pandas as pd

            sys.path.append(str(Path.cwd()))

            from tabular_model_utils import (
                build_tabular_dataset,
                classification_report_df,
                fit_and_evaluate_model,
                plot_evaluation_dashboard,
                plot_feature_importance,
                plot_mlp_learning_curve,
                set_seed,
            )
            """
        ),
        code(
            f"""
            MODEL_NAME = "{model_name}"
            FEATURE_GROUP = "eth_twitter_combined_features"
            ADD_GRAPH_STATS = False
            RANDOM_STATE = 42
            THRESHOLD_OBJECTIVE = "f1"

            set_seed(RANDOM_STATE)
            """
        ),
        code(
            """
            dataset = build_tabular_dataset(
                feature_group=FEATURE_GROUP,
                add_graph_stats=ADD_GRAPH_STATS,
                random_state=RANDOM_STATE,
            )

            display(dataset["split_df"])
            print("Number of input features:", len(dataset["feature_cols"]))
            print("Scale positive weight:", round(dataset["scale_pos_weight"], 4))
            """
        ),
        code(
            """
            result = fit_and_evaluate_model(
                model_name=MODEL_NAME,
                dataset=dataset,
                random_state=RANDOM_STATE,
                threshold_objective=THRESHOLD_OBJECTIVE,
            )

            metrics_df = pd.DataFrame([result["metrics"]]).T.rename(columns={0: "value"})
            display(metrics_df)
            """
        ),
        code(
            """
            report_df = classification_report_df(
                dataset["y_test"].to_numpy(),
                result["test_prob"],
                result["threshold"],
            )
            display(report_df)
            """
        ),
        code(
            """
            plot_evaluation_dashboard(
                y_true=dataset["y_test"].to_numpy(),
                y_prob=result["test_prob"],
                threshold=result["threshold"],
                title_prefix=MODEL_NAME.replace("_", " ").title(),
            )
            """
        ),
        code(
            """
            test_node_rows = dataset["nodes_df"].loc[dataset["test_idx"]].copy()
            test_node_rows["predicted_probability"] = result["test_prob"]
            test_node_rows["predicted_label"] = (result["test_prob"] >= result["threshold"]).astype(int)
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
    ] + (
        [
            code(
                """
                plot_feature_importance(
                    model=result["model"],
                    feature_cols=dataset["feature_cols"],
                    title=f"{MODEL_NAME.replace('_', ' ').title()} Feature Importance",
                    top_n=20,
                )
                """
            )
        ]
        if importance_supported
        else []
    ) + (
        [
            code(
                """
                plot_mlp_learning_curve(result["model"])
                """
            )
        ]
        if include_learning_curve
        else []
    ) + [
        md(
            """
            ## Notes

            Use `PR-AUC`, `Recall`, `Precision`, `F1`, `Balanced-Accuracy`, and `MCC` together when comparing this model with the other baselines.
            """
        )
    ]


comparison_cells = [
    md(
        """
        # 05. Model Comparison

        This notebook runs all four tabular baselines under the same split and feature setup:

        - Logistic Regression
        - Random Forest
        - MLP
        - XGBoost
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import pandas as pd

        sys.path.append(str(Path.cwd()))

        from tabular_model_utils import (
            build_tabular_dataset,
            fit_and_evaluate_model,
            plot_comparison_bars,
            plot_evaluation_dashboard,
            set_seed,
        )
        """
    ),
    code(
        """
        FEATURE_GROUP = "eth_twitter_combined_features"
        ADD_GRAPH_STATS = False
        RANDOM_STATE = 42
        THRESHOLD_OBJECTIVE = "f1"
        MODELS = [
            "logistic_regression",
            "random_forest",
            "mlp",
            "xgboost",
        ]

        set_seed(RANDOM_STATE)
        """
    ),
    code(
        """
        dataset = build_tabular_dataset(
            feature_group=FEATURE_GROUP,
            add_graph_stats=ADD_GRAPH_STATS,
            random_state=RANDOM_STATE,
        )
        display(dataset["split_df"])
        print("Number of input features:", len(dataset["feature_cols"]))
        """
    ),
    code(
        """
        results = []
        fitted = {}

        for model_name in MODELS:
            result = fit_and_evaluate_model(
                model_name=model_name,
                dataset=dataset,
                random_state=RANDOM_STATE,
                threshold_objective=THRESHOLD_OBJECTIVE,
            )
            fitted[model_name] = result
            results.append({"model": model_name, **result["metrics"]})

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
        print("Best model:", best_model_name)
        best_result = fitted[best_model_name]

        plot_evaluation_dashboard(
            y_true=dataset["y_test"].to_numpy(),
            y_prob=best_result["test_prob"],
            threshold=best_result["threshold"],
            title_prefix=best_model_name.replace("_", " ").title(),
        )
        """
    ),
    code(
        """
        comparison_cols = [
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
        ]
        results_df[comparison_cols]
        """
    ),
    md(
        """
        ## Notes

        This notebook is the main leaderboard for the non-graph baselines.
        It defaults to a stricter feature-only setup by keeping `ADD_GRAPH_STATS = False`.
        Keep the graph-model comparisons separate from this table unless the split protocol and input features are aligned.
        """
    ),
]


write_notebook(TABULAR_NOTEBOOKS / "00_tabular_data_overview.ipynb", overview_cells)
write_notebook(
    TABULAR_NOTEBOOKS / "01_logistic_regression.ipynb",
    model_notebook_cells("logistic_regression", "01. Logistic Regression", importance_supported=True),
)
write_notebook(
    TABULAR_NOTEBOOKS / "02_random_forest.ipynb",
    model_notebook_cells("random_forest", "02. Random Forest", importance_supported=True),
)
write_notebook(
    TABULAR_NOTEBOOKS / "03_mlp.ipynb",
    model_notebook_cells("mlp", "03. MLP", importance_supported=False, include_learning_curve=True),
)
write_notebook(
    TABULAR_NOTEBOOKS / "04_xgboost.ipynb",
    model_notebook_cells("xgboost", "04. XGBoost", importance_supported=True),
)
write_notebook(TABULAR_NOTEBOOKS / "05_model_comparison.ipynb", comparison_cells)

print(f"Wrote tabular model notebooks to {TABULAR_NOTEBOOKS}")
