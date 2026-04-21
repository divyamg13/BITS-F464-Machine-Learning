from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


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


nb1_cells = [
    md(
        """
        # 01. Data Overview And EDA

        This notebook is the project starting point for the denser `10,000` node sample:

        - `wash_trading_gnn_nodes_10000.csv`
        - `wash_trading_gnn_edges_10000.csv`

        It answers four questions before any modeling:

        1. What is the class balance?
        2. How connected is the sampled graph?
        3. Which feature groups are available?
        4. Are there missing values or feature issues that need to be handled downstream?
        """
    ),
    code(
        """
        from pathlib import Path

        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from sklearn.model_selection import train_test_split

        sns.set_theme(style="whitegrid")

        ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
        NODES_PATH = ROOT / "wash_trading_gnn_nodes_10000.csv"
        EDGES_PATH = ROOT / "wash_trading_gnn_edges_10000.csv"

        nodes_df = pd.read_csv(NODES_PATH)
        edges_df = pd.read_csv(EDGES_PATH)

        print(f"Nodes path: {NODES_PATH}")
        print(f"Edges path: {EDGES_PATH}")
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
        feature_groups = {
            "features": [c for c in nodes_df.columns if c.startswith("features_")],
            "normalized_log_features": [c for c in nodes_df.columns if c.startswith("normalized_log_features_")],
            "twitter_semantic_features": [c for c in nodes_df.columns if c.startswith("twitter_semantic_features_")],
            "twitter_semantic_features_normalized": [c for c in nodes_df.columns if c.startswith("twitter_semantic_features_normalized_")],
            "twitter_deepwalk_features": [c for c in nodes_df.columns if c.startswith("twitter_deepwalk_features_")],
            "twitter_deepwalk_normalized_features": [c for c in nodes_df.columns if c.startswith("twitter_deepwalk_normalized_features_")],
            "twitter_combined_features": [c for c in nodes_df.columns if c.startswith("twitter_combined_features_")],
            "eth_twitter_combined_features": [c for c in nodes_df.columns if c.startswith("eth_twitter_combined_features_")],
            "graph_stats": [
                "full_in_degree",
                "full_out_degree",
                "full_total_degree",
                "full_positive_touch_count",
                "full_has_self_loop",
                "sub_in_degree",
                "sub_out_degree",
                "sub_total_degree",
            ],
        }

        pd.Series({name: len(cols) for name, cols in feature_groups.items()}).sort_values(ascending=False)
        """
    ),
    code(
        """
        label_counts = nodes_df["label"].value_counts().sort_index()
        label_ratio = (label_counts / label_counts.sum()).rename("ratio")
        display(pd.concat([label_counts.rename("count"), label_ratio], axis=1))

        plt.figure(figsize=(5, 4))
        sns.countplot(data=nodes_df, x="label")
        plt.title("Label Distribution")
        plt.show()
        """
    ),
    code(
        """
        G = nx.from_pandas_edgelist(
            edges_df,
            source="src_node_id",
            target="dst_node_id",
            create_using=nx.DiGraph(),
        )

        connected_nodes = set(edges_df["src_node_id"]).union(edges_df["dst_node_id"])
        isolated_count = int((~nodes_df["node_id"].isin(connected_nodes)).sum())

        graph_summary = {
            "num_nodes_in_table": len(nodes_df),
            "num_edges_in_table": len(edges_df),
            "graph_nodes": G.number_of_nodes(),
            "graph_edges": G.number_of_edges(),
            "isolated_nodes": isolated_count,
            "density": nx.density(G),
            "self_loops": nx.number_of_selfloops(G),
        }

        pd.Series(graph_summary)
        """
    ),
    code(
        """
        component_sizes = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
        pd.Series({
            "num_weak_components": len(component_sizes),
            "largest_component_size": component_sizes[0],
            "median_component_size": float(np.median(component_sizes)),
            "top_10_component_sizes": component_sizes[:10],
        })
        """
    ),
    code(
        """
        nan_counts = nodes_df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        print(f"Columns with NaNs: {len(nan_counts)}")
        display(nan_counts.head(20))
        """
    ),
    code(
        """
        split_seed = 42

        train_ids, temp_ids = train_test_split(
            nodes_df["node_id"],
            test_size=0.30,
            random_state=split_seed,
            stratify=nodes_df["label"],
        )

        temp_df = nodes_df[nodes_df["node_id"].isin(temp_ids)]
        val_ids, test_ids = train_test_split(
            temp_df["node_id"],
            test_size=0.50,
            random_state=split_seed,
            stratify=temp_df["label"],
        )

        split_summary = []
        for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            part = nodes_df[nodes_df["node_id"].isin(split_ids)]
            split_summary.append(
                {
                    "split": split_name,
                    "rows": len(part),
                    "positives": int(part["label"].sum()),
                    "positive_ratio": float(part["label"].mean()),
                }
            )

        pd.DataFrame(split_summary)
        """
    ),
    md(
        """
        ## Next

        - `02_tabular_baselines.ipynb`: feature-only supervised baselines
        - `03_graph_feature_baselines.ipynb`: add degree/PageRank/component features
        - `04_gnn_models.ipynb`: GraphSAGE, GAT, or GGNN on the sampled graph
        """
    ),
]


nb2_cells = [
    md(
        """
        # 02. Tabular Baselines

        This notebook treats the task as wallet-level supervised classification and ignores the edge list.

        Recommended default:

        - feature block: `eth_twitter_combined_features_*`
        - target: `label`
        - main metric: `PR-AUC`
        """
    ),
    code(
        """
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
        nodes_df = pd.read_csv(ROOT / "wash_trading_gnn_nodes_10000.csv")
        """
    ),
    code(
        """
        FEATURE_GROUP = "eth_twitter_combined_features"
        RANDOM_STATE = 42

        feature_groups = {
            "features": [c for c in nodes_df.columns if c.startswith("features_")],
            "normalized_log_features": [c for c in nodes_df.columns if c.startswith("normalized_log_features_")],
            "twitter_combined_features": [c for c in nodes_df.columns if c.startswith("twitter_combined_features_")],
            "eth_twitter_combined_features": [c for c in nodes_df.columns if c.startswith("eth_twitter_combined_features_")],
        }

        feature_cols = feature_groups[FEATURE_GROUP]
        X = nodes_df[feature_cols].copy()
        y = nodes_df["label"].copy()

        train_idx, temp_idx = train_test_split(
            nodes_df.index,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y.loc[temp_idx],
        )

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        print(X_train.shape, X_val.shape, X_test.shape)
        """
    ),
    code(
        """
        def best_threshold(y_true, y_prob, thresholds=None):
            thresholds = np.linspace(0.05, 0.95, 37) if thresholds is None else thresholds
            best = {"threshold": 0.5, "f1": -1.0}
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best["f1"]:
                    best = {"threshold": float(threshold), "f1": float(f1)}
            return best["threshold"]


        def evaluate_binary_classifier(model_name, y_true, y_prob, threshold):
            y_pred = (y_prob >= threshold).astype(int)
            return {
                "model": model_name,
                "threshold": threshold,
                "PR-AUC": average_precision_score(y_true, y_prob),
                "ROC-AUC": roc_auc_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "Balanced-Accuracy": balanced_accuracy_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "Accuracy": accuracy_score(y_true, y_pred),
            }
        """
    ),
    code(
        """
        scale_pos_weight = y_train.eq(0).sum() / y_train.eq(1).sum()

        models = {
            "logreg": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            ),
            "random_forest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "mlp": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            learning_rate_init=1e-3,
                            max_iter=200,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "xgboost": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            scale_pos_weight=scale_pos_weight,
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        }
        """
    ),
    code(
        """
        results = []
        fitted_models = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            threshold = best_threshold(y_val.to_numpy(), val_prob)
            test_prob = model.predict_proba(X_test)[:, 1]
            results.append(evaluate_binary_classifier(model_name, y_test.to_numpy(), test_prob, threshold))
            fitted_models[model_name] = model

        results_df = pd.DataFrame(results).sort_values(["PR-AUC", "F1"], ascending=False)
        display(results_df)
        """
    ),
    code(
        """
        best_model_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_model_name]
        best_model
        """
    ),
    md(
        """
        ## Notes

        - Keep `PR-AUC` as the primary leaderboard metric.
        - `XGBoost` is usually the strongest baseline here.
        - The next notebook adds graph-derived features to test whether message passing is actually necessary.
        """
    ),
]


nb3_cells = [
    md(
        """
        # 03. Graph-Feature Baselines

        This notebook sits between plain tabular modeling and full GNNs.

        It computes simple structural features from the sampled edge list and fuses them with the compact wallet feature block.
        """
    ),
    code(
        """
        from pathlib import Path

        import networkx as nx
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
        nodes_df = pd.read_csv(ROOT / "wash_trading_gnn_nodes_10000.csv")
        edges_df = pd.read_csv(ROOT / "wash_trading_gnn_edges_10000.csv")
        """
    ),
    code(
        """
        G = nx.from_pandas_edgelist(
            edges_df,
            source="src_node_id",
            target="dst_node_id",
            create_using=nx.DiGraph(),
        )

        pagerank = nx.pagerank(G, alpha=0.85)
        weak_component_size = {}
        for component in nx.weakly_connected_components(G):
            size = len(component)
            for node in component:
                weak_component_size[node] = size

        graph_features = pd.DataFrame(
            {
                "node_id": nodes_df["node_id"],
                "nx_in_degree": nodes_df["node_id"].map(dict(G.in_degree())).fillna(0).astype(float),
                "nx_out_degree": nodes_df["node_id"].map(dict(G.out_degree())).fillna(0).astype(float),
                "nx_total_degree": nodes_df["node_id"].map(dict(G.degree())).fillna(0).astype(float),
                "nx_pagerank": nodes_df["node_id"].map(pagerank).fillna(0.0),
                "nx_component_size": nodes_df["node_id"].map(weak_component_size).fillna(1).astype(float),
                "nx_has_self_loop": nodes_df["node_id"].isin(nx.nodes_with_selfloops(G)).astype(int),
            }
        )

        display(graph_features.head())
        """
    ),
    code(
        """
        fused_df = nodes_df.merge(graph_features, on="node_id", how="left")

        base_features = [c for c in fused_df.columns if c.startswith("eth_twitter_combined_features_")]
        graph_stats = [
            "full_in_degree",
            "full_out_degree",
            "full_total_degree",
            "full_positive_touch_count",
            "full_has_self_loop",
            "sub_in_degree",
            "sub_out_degree",
            "sub_total_degree",
            "nx_in_degree",
            "nx_out_degree",
            "nx_total_degree",
            "nx_pagerank",
            "nx_component_size",
            "nx_has_self_loop",
        ]

        X_base = fused_df[base_features].copy()
        X_fused = fused_df[base_features + graph_stats].copy()
        y = fused_df["label"].copy()
        """
    ),
    code(
        """
        def best_threshold(y_true, y_prob):
            thresholds = np.linspace(0.05, 0.95, 37)
            best_t, best_f1 = 0.5, -1.0
            for threshold in thresholds:
                score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
                if score > best_f1:
                    best_t, best_f1 = float(threshold), float(score)
            return best_t


        def evaluate(name, y_true, y_prob, threshold):
            y_pred = (y_prob >= threshold).astype(int)
            return {
                "model": name,
                "threshold": threshold,
                "PR-AUC": average_precision_score(y_true, y_prob),
                "ROC-AUC": roc_auc_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "Balanced-Accuracy": balanced_accuracy_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "Accuracy": accuracy_score(y_true, y_pred),
            }
        """
    ),
    code(
        """
        RANDOM_STATE = 42
        train_idx, temp_idx = train_test_split(
            fused_df.index,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y.loc[temp_idx],
        )

        y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
        scale_pos_weight = y_train.eq(0).sum() / y_train.eq(1).sum()
        """
    ),
    code(
        """
        def run_xgb(X_train, X_val, X_test):
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            scale_pos_weight=scale_pos_weight,
                            tree_method="hist",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            threshold = best_threshold(y_val.to_numpy(), val_prob)
            return evaluate("xgboost", y_test.to_numpy(), test_prob, threshold)


        def run_logreg(X_train, X_val, X_test):
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]
            )
            model.fit(X_train, y_train)
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
            threshold = best_threshold(y_val.to_numpy(), val_prob)
            return evaluate("logreg", y_test.to_numpy(), test_prob, threshold)
        """
    ),
    code(
        """
        comparison_rows = []
        for dataset_name, X_data in [("node_features_only", X_base), ("node_plus_graph_stats", X_fused)]:
            comparison_rows.append({"dataset": dataset_name, **run_logreg(X_data.loc[train_idx], X_data.loc[val_idx], X_data.loc[test_idx])})
            comparison_rows.append({"dataset": dataset_name, **run_xgb(X_data.loc[train_idx], X_data.loc[val_idx], X_data.loc[test_idx])})

        comparison_df = pd.DataFrame(comparison_rows).sort_values(["PR-AUC", "F1"], ascending=False)
        display(comparison_df)
        """
    ),
    md(
        """
        ## Interpretation

        If `node_plus_graph_stats` clearly beats `node_features_only`, then graph structure is already helpful even before a GNN.
        That makes the GNN comparison meaningful instead of decorative.
        """
    ),
]


nb4_cells = [
    md(
        """
        # 04. GNN Models

        This notebook builds a single-node classification pipeline on the `10k` sampled graph.

        Supported models:

        - `graphsage`
        - `gat`
        - `ggnn`

        Recommended first run on CPU:

        - `MODEL_NAME = "graphsage"`
        - `FEATURE_GROUP = "eth_twitter_combined_features"`
        - `EPOCHS = 80`
        """
    ),
    code(
        """
        from pathlib import Path
        import copy
        import random

        import dgl
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from dgl.nn import GATConv, GatedGraphConv, SAGEConv
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split

        ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
        nodes_df = pd.read_csv(ROOT / "wash_trading_gnn_nodes_10000.csv")
        edges_df = pd.read_csv(ROOT / "wash_trading_gnn_edges_10000.csv")
        """
    ),
    code(
        """
        MODEL_NAME = "graphsage"  # one of: graphsage, gat, ggnn
        FEATURE_GROUP = "eth_twitter_combined_features"
        ADD_GRAPH_STATS = True
        RANDOM_STATE = 42
        HIDDEN_DIM = 64
        DROPOUT = 0.2
        LEARNING_RATE = 1e-3
        WEIGHT_DECAY = 1e-4
        EPOCHS = 80
        PATIENCE = 15

        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        """
    ),
    code(
        """
        feature_groups = {
            "features": [c for c in nodes_df.columns if c.startswith("features_")],
            "normalized_log_features": [c for c in nodes_df.columns if c.startswith("normalized_log_features_")],
            "twitter_combined_features": [c for c in nodes_df.columns if c.startswith("twitter_combined_features_")],
            "eth_twitter_combined_features": [c for c in nodes_df.columns if c.startswith("eth_twitter_combined_features_")],
        }
        graph_stat_cols = [
            "full_in_degree",
            "full_out_degree",
            "full_total_degree",
            "full_positive_touch_count",
            "full_has_self_loop",
            "sub_in_degree",
            "sub_out_degree",
            "sub_total_degree",
        ]

        feature_cols = feature_groups[FEATURE_GROUP] + (graph_stat_cols if ADD_GRAPH_STATS else [])
        print("Number of input features:", len(feature_cols))
        """
    ),
    code(
        """
        node_ids = nodes_df["node_id"].tolist()
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        src = edges_df["src_node_id"].map(node_to_idx).to_numpy()
        dst = edges_df["dst_node_id"].map(node_to_idx).to_numpy()

        graph = dgl.graph((src, dst), num_nodes=len(nodes_df))
        graph = dgl.add_self_loop(graph)

        x = torch.tensor(nodes_df[feature_cols].fillna(0.0).to_numpy(), dtype=torch.float32)
        y = torch.tensor(nodes_df["label"].to_numpy(), dtype=torch.long)

        train_idx, temp_idx = train_test_split(
            np.arange(len(nodes_df)),
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=nodes_df["label"],
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=nodes_df.iloc[temp_idx]["label"],
        )

        train_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
        val_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
        test_mask = torch.zeros(len(nodes_df), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_mean = x[train_mask].mean(0, keepdim=True)
        train_std = x[train_mask].std(0, keepdim=True).clamp_min(1e-6)
        x = (x - train_mean) / train_std

        class_counts = torch.bincount(y[train_mask])
        class_weights = class_counts.sum() / (len(class_counts) * class_counts.float())
        class_weights
        """
    ),
    code(
        """
        class GraphSAGEModel(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim, dropout):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hidden_dim, "mean")
                self.conv2 = SAGEConv(hidden_dim, hidden_dim, "mean")
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
            def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads=4):
                super().__init__()
                self.gat1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
                self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads=1, feat_drop=dropout, attn_drop=dropout)
                self.classifier = nn.Linear(hidden_dim, out_dim)

            def forward(self, g, features):
                h = self.gat1(g, features).flatten(1)
                h = F.elu(h)
                h = self.gat2(g, h).squeeze(1)
                h = F.elu(h)
                return self.classifier(h)


        class GGNNModel(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim, n_steps=3):
                super().__init__()
                self.input_proj = nn.Linear(in_dim, hidden_dim)
                self.ggnn = GatedGraphConv(hidden_dim, hidden_dim, n_steps=n_steps, n_etypes=1)
                self.classifier = nn.Linear(hidden_dim, out_dim)

            def forward(self, g, features):
                h = self.input_proj(features)
                etypes = torch.zeros(g.num_edges(), dtype=torch.long)
                h = self.ggnn(g, h, etypes)
                h = F.relu(h)
                return self.classifier(h)


        if MODEL_NAME == "graphsage":
            model = GraphSAGEModel(x.shape[1], HIDDEN_DIM, 2, DROPOUT)
        elif MODEL_NAME == "gat":
            model = GATModel(x.shape[1], HIDDEN_DIM, 2, DROPOUT)
        elif MODEL_NAME == "ggnn":
            model = GGNNModel(x.shape[1], HIDDEN_DIM, 2)
        else:
            raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

        model
        """
    ),
    code(
        """
        def find_best_threshold(y_true, y_prob):
            thresholds = np.linspace(0.05, 0.95, 37)
            best_threshold = 0.5
            best_f1 = -1.0
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_f1:
                    best_threshold = float(threshold)
                    best_f1 = float(score)
            return best_threshold


        def compute_metrics(y_true, y_prob, threshold):
            y_pred = (y_prob >= threshold).astype(int)
            return {
                "threshold": threshold,
                "PR-AUC": average_precision_score(y_true, y_prob),
                "ROC-AUC": roc_auc_score(y_true, y_prob),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "Balanced-Accuracy": balanced_accuracy_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "Accuracy": accuracy_score(y_true, y_pred),
            }
        """
    ),
    code(
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_state = None
        best_val_pr_auc = -1.0
        patience_left = PATIENCE
        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            logits = model(graph, x)
            loss = criterion(logits[train_mask], y[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(graph, x)
                val_prob = torch.softmax(logits[val_mask], dim=1)[:, 1].cpu().numpy()
                val_true = y[val_mask].cpu().numpy()
                val_pr_auc = average_precision_score(val_true, val_prob)
                val_threshold = find_best_threshold(val_true, val_prob)
                val_metrics = compute_metrics(val_true, val_prob, val_threshold)

            history.append({"epoch": epoch, "train_loss": float(loss.item()), **{f"val_{k}": v for k, v in val_metrics.items()}})

            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = float(val_pr_auc)
                best_state = copy.deepcopy(model.state_dict())
                patience_left = PATIENCE
            else:
                patience_left -= 1

            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train_loss={loss.item():.4f} | val_pr_auc={val_pr_auc:.4f} | val_f1={val_metrics['F1']:.4f}")

            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}")
                break
        """
    ),
    code(
        """
        history_df = pd.DataFrame(history)
        display(history_df.tail())
        """
    ),
    code(
        """
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            logits = model(graph, x)
            val_prob = torch.softmax(logits[val_mask], dim=1)[:, 1].cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            best_threshold = find_best_threshold(val_true, val_prob)

            test_prob = torch.softmax(logits[test_mask], dim=1)[:, 1].cpu().numpy()
            test_true = y[test_mask].cpu().numpy()
            test_metrics = compute_metrics(test_true, test_prob, best_threshold)

        print(f"Model: {MODEL_NAME}")
        pd.Series(test_metrics)
        """
    ),
    md(
        """
        ## Suggested usage

        1. Run `graphsage` first.
        2. Re-run with `gat`.
        3. Re-run with `ggnn`.
        4. Compare all three against notebook 2 and notebook 3 using `PR-AUC`, `F1`, and `Recall`.
        """
    ),
]


write_notebook(NOTEBOOKS / "01_data_overview.ipynb", nb1_cells)
write_notebook(NOTEBOOKS / "02_tabular_baselines.ipynb", nb2_cells)
write_notebook(NOTEBOOKS / "03_graph_feature_baselines.ipynb", nb3_cells)
write_notebook(NOTEBOOKS / "04_gnn_models.ipynb", nb4_cells)

print(f"Wrote notebooks to {NOTEBOOKS}")
