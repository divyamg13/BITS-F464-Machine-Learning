"""
Self-contained unsupervised anomaly detection + plot generation.

Trains 4 models (Isolation Forest, LOF, DBSCAN, Autoencoder) on wash-trading
node features WITHOUT using labels. Generates all evaluation dashboards and
comparison plots into unsupervised_results/.

No external project files needed — everything is inlined here.

Run:
    cd BITS-F464-Machine-Learning
    source ../mymlenv/bin/activate
    python unsupervised-approach/generate_all_plots.py

    # skip autoencoder (instant run):
    python unsupervised-approach/generate_all_plots.py --no-autoencoder
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file; no display window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── paths ─────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent          # unsupervised-approach/
ROOT        = HERE.parent                              # project root
NODES_CSV   = ROOT / "datasets" / "wash_trading_gnn_nodes_10000.csv"
if not NODES_CSV.exists():
    NODES_CSV = ROOT / "wash_trading_gnn_nodes_10000.csv"

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE  = 42
CONTAMINATION = 0.113   # ~true wash-trader rate


# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_and_split(nodes_csv: Path, random_state: int = 42):
    """Load nodes CSV, extract eth_twitter_combined_features, split 70/15/15."""
    df = pd.read_csv(nodes_csv)
    feature_cols = [c for c in df.columns if c.startswith("eth_twitter_combined_features_")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    # 70 / 30
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    # 30 → 15 / 15
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=random_state, stratify=y_tmp
    )

    print(f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} | Features={X.shape[1]}")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  {name}: {yy.sum()} positives ({yy.mean()*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                        objective: str = "f1") -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        score = (f1_score(y_true, y_pred, zero_division=0) if objective == "f1"
                 else recall_score(y_true, y_pred, zero_division=0))
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    npv        = tn / max(1, tn + fn)

    positive_count = int(y_true.sum())
    top_k = max(1, positive_count)
    ranked = np.argsort(-y_prob)[:top_k]
    selected = y_true[ranked]

    return {
        "Threshold":          threshold,
        "PR-AUC":             average_precision_score(y_true, y_prob),
        "ROC-AUC":            roc_auc_score(y_true, y_prob),
        "F1":                 f1_score(y_true, y_pred, zero_division=0),
        "Precision":          precision_score(y_true, y_pred, zero_division=0),
        "Recall":             recall_score(y_true, y_pred, zero_division=0),
        "Specificity":        specificity,
        "NPV":                npv,
        "Balanced-Accuracy":  balanced_accuracy_score(y_true, y_pred),
        "MCC":                matthews_corrcoef(y_true, y_pred),
        "Cohen-Kappa":        cohen_kappa_score(y_true, y_pred),
        "Brier-Score":        brier_score_loss(y_true, y_prob),
        "Log-Loss":           log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "Accuracy":           accuracy_score(y_true, y_pred),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "TopK":               top_k,
        "Precision@K":        float(selected.mean()),
        "Recall@K":           float(selected.sum() / max(1, positive_count)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PLOT: 6-PANEL EVALUATION DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def plot_evaluation_dashboard(y_true: np.ndarray, y_prob: np.ndarray,
                              threshold: float, title_prefix: str = "") -> plt.Figure:
    """
    6-panel evaluation dashboard:
      [0] ROC curve          [1] Precision-Recall curve
      [2] Confusion matrix   [3] Score distribution by true label
      [4] Metrics vs threshold sweep   [5] Calibration curve
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _                  = roc_curve(y_true, y_prob)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                                 strategy="quantile")

    # threshold sweep
    thr_grid = np.linspace(0.05, 0.95, 37)
    sweep = []
    for t in thr_grid:
        yp = (y_prob >= t).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        sweep.append({
            "threshold":   t,
            "precision":   precision_score(y_true, yp, zero_division=0),
            "recall":      recall_score(y_true, yp, zero_division=0),
            "f1":          f1_score(y_true, yp, zero_division=0),
            "specificity": _tn / max(1, _tn + _fp),
        })
    sweep_df = pd.DataFrame(sweep)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    prefix = f"{title_prefix} " if title_prefix else ""

    # ── Panel 0: ROC ──────────────────────────────────────────────────────────
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.4f}", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_title(f"{prefix}ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend()

    # ── Panel 1: PR curve ─────────────────────────────────────────────────────
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(recall_arr, precision_arr, label=f"PR-AUC = {ap:.4f}", lw=2)
    axes[1].axhline(y_true.mean(), color="gray", linestyle="--", lw=1,
                    label=f"Random baseline = {y_true.mean():.3f}")
    axes[1].set_title(f"{prefix}Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    # ── Panel 2: Confusion matrix ─────────────────────────────────────────────
    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
        xticklabels=["Pred: Normal", "Pred: Wash Trader"],
        yticklabels=["True: Normal", "True: Wash Trader"],
    )
    axes[2].set_title(f"{prefix}Confusion Matrix @ threshold={threshold:.2f}")

    # ── Panel 3: Score distribution ───────────────────────────────────────────
    axes[3].hist(y_prob[y_true == 0], bins=40, alpha=0.65, label="Normal (label=0)",
                 color="steelblue")
    axes[3].hist(y_prob[y_true == 1], bins=40, alpha=0.65, label="Wash Trader (label=1)",
                 color="tomato")
    axes[3].axvline(threshold, color="black", linestyle="--", lw=1.5,
                    label=f"Threshold = {threshold:.2f}")
    axes[3].set_title(f"{prefix}Anomaly Score Distribution")
    axes[3].set_xlabel("Normalised anomaly score  (higher = more suspicious)")
    axes[3].legend()

    # ── Panel 4: Threshold sweep ──────────────────────────────────────────────
    axes[4].plot(sweep_df["threshold"], sweep_df["precision"],  label="Precision")
    axes[4].plot(sweep_df["threshold"], sweep_df["recall"],     label="Recall")
    axes[4].plot(sweep_df["threshold"], sweep_df["f1"],         label="F1")
    axes[4].plot(sweep_df["threshold"], sweep_df["specificity"],label="Specificity")
    axes[4].axvline(threshold, color="black", linestyle="--", lw=1.5,
                    label=f"Chosen = {threshold:.2f}")
    axes[4].set_title(f"{prefix}Metrics vs Decision Threshold")
    axes[4].set_xlabel("Threshold")
    axes[4].legend(fontsize=8)

    # ── Panel 5: Calibration ──────────────────────────────────────────────────
    axes[5].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[5].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
    axes[5].set_title(f"{prefix}Calibration Curve")
    axes[5].set_xlabel("Mean predicted score")
    axes[5].set_ylabel("Fraction of actual positives")
    axes[5].legend()

    plt.suptitle(f"{prefix}Evaluation Dashboard", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SCORE UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def normalize_scores(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = hi - lo
    if denom == 0:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / denom, 0.0, 1.0)


def run_eval(name, y_val, y_test, val_raw, test_raw, train_raw, results):
    """Normalize scores, find threshold on val, evaluate on test, store result."""
    lo, hi  = train_raw.min(), train_raw.max()
    val_p   = normalize_scores(val_raw,  lo, hi)
    test_p  = normalize_scores(test_raw, lo, hi)
    thr     = find_best_threshold(y_val, val_p)
    metrics = evaluate_probabilities(y_test, test_p, thr)
    results.append({"model": name, **metrics})
    print(f"  PR-AUC={metrics['PR-AUC']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
          f"F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}  "
          f"Precision={metrics['Precision']:.4f}")
    return test_p, thr


def save_dashboard(y_test, test_p, thr, name, fname):
    fig = plot_evaluation_dashboard(y_test, test_p, thr, title_prefix=name)
    fig.savefig(RESULTS_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → unsupervised_results/{fname}")


# ═════════════════════════════════════════════════════════════════════════════
# DBSCAN HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def auto_eps(X: np.ndarray, min_samples: int) -> float:
    """15th percentile of min_samples-th NN distances → tight density threshold."""
    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    return float(np.percentile(dists[:, -1], 15))


def fit_dbscan(X_train: np.ndarray, min_samples: int = 10):
    eps = auto_eps(X_train, min_samples)
    db  = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X_train)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise    = int((db.labels_ == -1).sum())
    print(f"  eps={eps:.4f}  clusters={n_clusters}  "
          f"noise={n_noise} ({n_noise/len(X_train)*100:.1f}% of train)")
    if len(db.core_sample_indices_) == 0:
        return None, eps, n_clusters, n_noise
    core_nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
        X_train[db.core_sample_indices_]
    )
    return core_nbrs, eps, n_clusters, n_noise


def dbscan_score(core_nbrs, X: np.ndarray) -> np.ndarray:
    if core_nbrs is None:
        return np.ones(len(X), dtype=float)
    dists, _ = core_nbrs.kneighbors(X)
    return dists[:, 0]


# ═════════════════════════════════════════════════════════════════════════════
# AUTOENCODER
# ═════════════════════════════════════════════════════════════════════════════

def build_autoencoder(input_dim: int, bottleneck: int = 8):
    h = input_dim * 2
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h), nn.ReLU(),
                nn.Linear(h, bottleneck), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, h), nn.ReLU(),
                nn.Linear(h, input_dim),
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))
        @torch.no_grad()
        def reconstruction_error(self, x):
            return ((x - self.forward(x)) ** 2).mean(dim=1)
    return AE()


def train_autoencoder(X_scaled, epochs, device, batch_size=64, lr=1e-3):
    model = build_autoencoder(X_scaled.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()
    X_t   = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t), batch_size=batch_size, shuffle=True
    )
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        avg = total / len(X_t)
        history.append(avg)
        if epoch == 1 or epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg:.6f}")
    return model, history


def ae_scores(model, X_scaled, device) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    return model.reconstruction_error(X_t).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOT
# ═════════════════════════════════════════════════════════════════════════════

def plot_comparison(results_df: pd.DataFrame) -> None:
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision", "MCC"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=results_df, x="model", y=metric, hue="model",
                    ax=ax, palette="Set2", legend=False)
        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(1.0, results_df[metric].max() * 1.15))
        ax.set_xlabel("")
    plt.suptitle("Unsupervised Model Comparison (test set)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "unsupervised_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → unsupervised_results/unsupervised_comparison.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(run_autoencoder: bool = True, ae_epochs: int = 100, device_str: str = "auto"):
    set_seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid")

    run_autoencoder = run_autoencoder and TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        print("NOTE: torch not installed — autoencoder skipped. pip install torch to enable.\n")

    # ── device ────────────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        if device_str == "auto":
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("mps") if torch.backends.mps.is_available()
                      else torch.device("cpu"))
        else:
            device = torch.device(device_str)
        print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading data from {NODES_CSV.name}...")
    if not NODES_CSV.exists():
        sys.exit(f"ERROR: nodes CSV not found at {NODES_CSV}")
    X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = load_and_split(
        NODES_CSV, RANDOM_STATE
    )

    # ── preprocessing ─────────────────────────────────────────────────────────
    print("\n[2/7] Preprocessing (median impute → StandardScaler)...")
    pre       = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler",  StandardScaler())])
    X_train_s = pre.fit_transform(X_train)
    X_val_s   = pre.transform(X_val)
    X_test_s  = pre.transform(X_test)

    results = []

    # ── 1. Isolation Forest ───────────────────────────────────────────────────
    print("\n[3/7] Isolation Forest...")
    t0  = time.time()
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_s)
    iso_test_p, iso_thr = run_eval(
        "IsolationForest", y_val, y_test,
        val_raw=-iso.decision_function(X_val_s),
        test_raw=-iso.decision_function(X_test_s),
        train_raw=-iso.decision_function(X_train_s),
        results=results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, iso_test_p, iso_thr,
                   "Isolation Forest", "isolation_forest_dashboard.png")

    # ── 2. LOF ────────────────────────────────────────────────────────────────
    print("\n[4/7] Local Outlier Factor (LOF)...")
    t0  = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION,
                             novelty=True, n_jobs=-1)
    lof.fit(X_train_s)
    lof_test_p, lof_thr = run_eval(
        "LOF", y_val, y_test,
        val_raw=-lof.decision_function(X_val_s),
        test_raw=-lof.decision_function(X_test_s),
        train_raw=-lof.decision_function(X_train_s),
        results=results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, lof_test_p, lof_thr, "LOF", "lof_dashboard.png")

    # ── 3. DBSCAN ─────────────────────────────────────────────────────────────
    print("\n[5/7] DBSCAN...")
    t0 = time.time()
    MIN_SAMPLES = 10
    core_nbrs, eps_used, n_clusters, n_noise = fit_dbscan(X_train_s, MIN_SAMPLES)

    db_train_raw = dbscan_score(core_nbrs, X_train_s)
    db_val_raw   = dbscan_score(core_nbrs, X_val_s)
    db_test_raw  = dbscan_score(core_nbrs, X_test_s)

    db_test_p, db_thr = run_eval(
        "DBSCAN", y_val, y_test,
        val_raw=db_val_raw, test_raw=db_test_raw, train_raw=db_train_raw,
        results=results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, db_test_p, db_thr, "DBSCAN", "dbscan_dashboard.png")

    # k-distance diagnostic plot
    nbrs_k = NearestNeighbors(n_neighbors=MIN_SAMPLES, n_jobs=-1).fit(X_train_s)
    dists_k, _ = nbrs_k.kneighbors(X_train_s)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.sort(dists_k[:, -1]))
    ax.axhline(eps_used, color="red", linestyle="--",
               label=f"Chosen eps = {eps_used:.4f}")
    ax.set_title(f"DBSCAN k-distance plot  (k={MIN_SAMPLES})")
    ax.set_xlabel("Training points sorted by k-NN distance")
    ax.set_ylabel(f"{MIN_SAMPLES}-NN distance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "dbscan_kdistance_plot.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → unsupervised_results/dbscan_kdistance_plot.png")

    # ── 4. Autoencoder ────────────────────────────────────────────────────────
    if run_autoencoder:
        print(f"\n[6/7] Autoencoder ({ae_epochs} epochs on {device})...")
        t0 = time.time()
        ae_model, ae_history = train_autoencoder(X_train_s, ae_epochs, device)

        ae_test_p, ae_thr = run_eval(
            "Autoencoder", y_val, y_test,
            val_raw=ae_scores(ae_model, X_val_s, device),
            test_raw=ae_scores(ae_model, X_test_s, device),
            train_raw=ae_scores(ae_model, X_train_s, device),
            results=results,
        )
        print(f"  Time: {time.time()-t0:.1f}s")
        save_dashboard(y_test, ae_test_p, ae_thr,
                       "Autoencoder", "autoencoder_dashboard.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ae_history, color="steelblue")
        ax.set_title("Autoencoder Training Loss (MSE per epoch)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.grid(True)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "autoencoder_loss_curve.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("  Saved → unsupervised_results/autoencoder_loss_curve.png")
    else:
        reason = "(--no-autoencoder)" if TORCH_AVAILABLE else "(torch not installed)"
        print(f"\n[6/7] Autoencoder skipped {reason}.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[7/7] Summary...")
    results_df = pd.DataFrame(results)
    display_cols = ["model", "PR-AUC", "ROC-AUC", "F1", "Precision",
                    "Recall", "Balanced-Accuracy", "MCC", "Threshold"]
    print("\n" + results_df[display_cols].to_string(index=False))

    results_df.to_csv(RESULTS_DIR / "unsupervised_results.csv", index=False)
    print(f"\nFull metrics → unsupervised_results/unsupervised_results.csv")

    if len(results_df) > 1:
        plot_comparison(results_df)

    print(f"\nAll plots in: {RESULTS_DIR}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised anomaly detection + plot generation")
    parser.add_argument("--no-autoencoder", action="store_true",
                        help="Skip autoencoder — runs in ~3 seconds total")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Autoencoder training epochs (default: 100)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for autoencoder (default: auto-detect)")
    args = parser.parse_args()
    main(run_autoencoder=not args.no_autoencoder,
         ae_epochs=args.epochs,
         device_str=args.device)
