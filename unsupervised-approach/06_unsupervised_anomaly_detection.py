"""
Unsupervised anomaly detection for wash trading.

Four approaches trained WITHOUT using labels:
  1. Isolation Forest  - path-length anomaly scoring        (< 1 sec CPU)
  2. Local Outlier Factor - density-based scoring           (< 3 sec CPU)
  3. DBSCAN            - cluster noise-point scoring        (< 5 sec CPU)
  4. Autoencoder (PyTorch) - reconstruction-error scoring   (~2-4 min CPU)

All models produce anomaly scores normalized to [0,1] and evaluated using the
same 21-metric framework as every other model in this project.

Setup:
    source /Users/aditya-mene/Documents/Acads/3-2/ML/mymlenv/bin/activate
    # torch is now installed — run full pipeline:
    python tabular_models/06_unsupervised_anomaly_detection.py

    # skip autoencoder for a fast run:
    python tabular_models/06_unsupervised_anomaly_detection.py --no-autoencoder

Flags:
    --no-autoencoder   skip autoencoder (sklearn only, instant)
    --epochs N         autoencoder epochs (default: 100)
    --device cpu|cuda|mps  override device (default: auto-detect)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tabular_models"))

from tabular_model_utils import (
    build_tabular_dataset,
    evaluate_probabilities,
    find_best_threshold,
    plot_evaluation_dashboard,
    set_seed,
)

RESULTS_DIR = ROOT / "tabular_models" / "unsupervised_results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CONTAMINATION = 0.113


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN anomaly scoring
# ─────────────────────────────────────────────────────────────────────────────

def _auto_eps(X: np.ndarray, min_samples: int) -> float:
    """
    Pick eps using the k-distance heuristic:
    compute min_samples-th NN distance for every training point,
    sort them, take the 15th percentile as eps.

    Low percentile → tight density requirement → more noise points → more anomalies flagged.
    15th percentile is calibrated to roughly match the 11.3% contamination rate.
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    k_dists = np.sort(dists[:, -1])
    return float(np.percentile(k_dists, 15))


def fit_dbscan_scorer(
    X_train: np.ndarray,
    min_samples: int = 10,
    eps: float | None = None,
) -> tuple[NearestNeighbors, float, int, int, float]:
    """
    Fit DBSCAN on training data and return a core-sample nearest-neighbour
    index for scoring unseen data.

    Anomaly score = distance to nearest core sample.
    Points far from all core samples = noise / anomalies.

    Returns:
        core_nbrs   : fitted NearestNeighbors over core samples (for scoring)
        eps_used    : the eps value actually used
        n_clusters  : number of clusters found (excluding noise)
        n_noise     : number of training noise points
        noise_frac  : fraction of training points labelled noise
    """
    if eps is None:
        eps = _auto_eps(X_train, min_samples)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    db.fit(X_train)

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise    = int((db.labels_ == -1).sum())
    noise_frac = n_noise / len(X_train)

    if len(db.core_sample_indices_) == 0:
        # Degenerate: eps is too small, everything is noise.
        # Score every point as maximally anomalous (distance = 0 → we'll return 1 after norm).
        return None, eps, n_clusters, n_noise, noise_frac

    core_samples = X_train[db.core_sample_indices_]
    core_nbrs    = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(core_samples)
    return core_nbrs, eps, n_clusters, n_noise, noise_frac


def dbscan_score(core_nbrs: NearestNeighbors | None, X: np.ndarray) -> np.ndarray:
    """Distance to nearest core sample. Higher = more anomalous."""
    if core_nbrs is None:
        return np.ones(len(X), dtype=float)
    dists, _ = core_nbrs.kneighbors(X)
    return dists[:, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

def _build_autoencoder_class():
    class Autoencoder(nn.Module):
        """24 → 48 → 8 → 48 → 24. High reconstruction error = anomaly."""

        def __init__(self, input_dim: int = 24, bottleneck: int = 8):
            super().__init__()
            h = input_dim * 2
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

    return Autoencoder


def train_autoencoder(X_scaled, epochs, device, batch_size=64, lr=1e-3):
    AE = _build_autoencoder_class()
    model = AE(input_dim=X_scaled.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    X_t    = torch.tensor(X_scaled, dtype=torch.float32).to(device)
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


def ae_scores(model, X_scaled, device):
    model.eval()
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    return model.reconstruction_error(X_t).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Score normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_scores(scores, lo, hi):
    denom = hi - lo
    if denom == 0:
        return np.zeros_like(scores, dtype=float)
    return np.clip((scores - lo) / denom, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_and_log(name, y_val, y_test, val_raw, test_raw, train_raw, all_results):
    lo, hi   = train_raw.min(), train_raw.max()
    val_p    = normalize_scores(val_raw,  lo, hi)
    test_p   = normalize_scores(test_raw, lo, hi)
    thr      = find_best_threshold(y_val, val_p, objective="f1")
    metrics  = evaluate_probabilities(y_test, test_p, thr)
    all_results.append({"model": name, **metrics})
    print(f"  PR-AUC={metrics['PR-AUC']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
          f"F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}  "
          f"Precision={metrics['Precision']:.4f}")
    return test_p, thr, metrics


def save_dashboard(y_test, test_p, thr, name, fname):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_evaluation_dashboard(y_test, test_p, thr, title_prefix=name)
    plt.savefig(RESULTS_DIR / fname, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard → unsupervised_results/{fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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
    print("\n[1/7] Loading data...")
    import tabular_model_utils as _tmu
    nodes_path = ROOT / "datasets" / "wash_trading_gnn_nodes_10000.csv"
    if not nodes_path.exists():
        nodes_path = ROOT / "wash_trading_gnn_nodes_10000.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes CSV not found under {ROOT}/datasets/")
    _tmu.NODES_PATH = nodes_path

    dataset = build_tabular_dataset(
        feature_group="eth_twitter_combined_features",
        add_graph_stats=False,
        random_state=RANDOM_STATE,
    )
    print(dataset["split_df"].to_string(index=False))

    X_train = dataset["X_train"].values
    X_val   = dataset["X_val"].values
    X_test  = dataset["X_test"].values
    y_val   = dataset["y_val"].to_numpy()
    y_test  = dataset["y_test"].to_numpy()
    print(f"\nTrain={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} | Features={X_train.shape[1]}")

    # ── preprocessing ─────────────────────────────────────────────────────────
    print("\n[2/7] Preprocessing (median impute → StandardScaler)...")
    pre       = Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler",  StandardScaler())])
    X_train_s = pre.fit_transform(X_train)
    X_val_s   = pre.transform(X_val)
    X_test_s  = pre.transform(X_test)

    all_results = []

    # ── 1. Isolation Forest ───────────────────────────────────────────────────
    print("\n[3/7] Isolation Forest...")
    t0  = time.time()
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_s)
    iso_test_p, iso_thr, _ = run_and_log(
        "IsolationForest", y_val, y_test,
        val_raw   = -iso.decision_function(X_val_s),
        test_raw  = -iso.decision_function(X_test_s),
        train_raw = -iso.decision_function(X_train_s),
        all_results=all_results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, iso_test_p, iso_thr, "IsolationForest", "isolation_forest_dashboard.png")

    # ── 2. LOF ────────────────────────────────────────────────────────────────
    print("\n[4/7] Local Outlier Factor (LOF)...")
    t0  = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION,
                             novelty=True, n_jobs=-1)
    lof.fit(X_train_s)
    lof_test_p, lof_thr, _ = run_and_log(
        "LOF", y_val, y_test,
        val_raw   = -lof.decision_function(X_val_s),
        test_raw  = -lof.decision_function(X_test_s),
        train_raw = -lof.decision_function(X_train_s),
        all_results=all_results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, lof_test_p, lof_thr, "LOF", "lof_dashboard.png")

    # ── 3. DBSCAN ─────────────────────────────────────────────────────────────
    print("\n[5/7] DBSCAN...")
    t0 = time.time()
    MIN_SAMPLES = 10

    core_nbrs, eps_used, n_clusters, n_noise, noise_frac = fit_dbscan_scorer(
        X_train_s, min_samples=MIN_SAMPLES
    )
    print(f"  eps={eps_used:.4f}  clusters={n_clusters}  "
          f"noise_pts={n_noise} ({noise_frac*100:.1f}% of train)")

    db_train_raw = dbscan_score(core_nbrs, X_train_s)
    db_val_raw   = dbscan_score(core_nbrs, X_val_s)
    db_test_raw  = dbscan_score(core_nbrs, X_test_s)

    db_test_p, db_thr, _ = run_and_log(
        "DBSCAN", y_val, y_test,
        val_raw=db_val_raw, test_raw=db_test_raw, train_raw=db_train_raw,
        all_results=all_results,
    )
    print(f"  Time: {time.time()-t0:.2f}s")
    save_dashboard(y_test, db_test_p, db_thr, "DBSCAN", "dbscan_dashboard.png")

    # ── 3b. DBSCAN k-distance plot (diagnostic) ───────────────────────────────
    nbrs_diag = NearestNeighbors(n_neighbors=MIN_SAMPLES, n_jobs=-1).fit(X_train_s)
    dists_diag, _ = nbrs_diag.kneighbors(X_train_s)
    k_dists_sorted = np.sort(dists_diag[:, -1])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_dists_sorted)
    ax.axhline(eps_used, color="red", linestyle="--", label=f"chosen eps={eps_used:.3f}")
    ax.set_title(f"DBSCAN k-distance plot (k={MIN_SAMPLES})")
    ax.set_xlabel("Points sorted by k-NN distance")
    ax.set_ylabel(f"{MIN_SAMPLES}-NN distance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dbscan_kdistance_plot.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  k-distance plot → unsupervised_results/dbscan_kdistance_plot.png")

    # ── 4. Autoencoder ────────────────────────────────────────────────────────
    if run_autoencoder:
        print(f"\n[6/7] Autoencoder ({ae_epochs} epochs on {device})...")
        t0 = time.time()
        ae_model, ae_history = train_autoencoder(X_train_s, ae_epochs, device)

        ae_train_raw = ae_scores(ae_model, X_train_s, device)
        ae_val_raw   = ae_scores(ae_model, X_val_s,   device)
        ae_test_raw  = ae_scores(ae_model, X_test_s,  device)

        ae_test_p, ae_thr, _ = run_and_log(
            "Autoencoder", y_val, y_test,
            val_raw=ae_val_raw, test_raw=ae_test_raw, train_raw=ae_train_raw,
            all_results=all_results,
        )
        print(f"  Time: {time.time()-t0:.1f}s")
        save_dashboard(y_test, ae_test_p, ae_thr, "Autoencoder", "autoencoder_dashboard.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ae_history)
        ax.set_title("Autoencoder Training Loss (MSE)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "autoencoder_loss_curve.png", dpi=100, bbox_inches="tight")
        plt.close()
    else:
        reason = "(--no-autoencoder)" if TORCH_AVAILABLE else "(torch not installed)"
        print(f"\n[6/7] Autoencoder skipped {reason}.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[7/7] Summary...")
    results_df = pd.DataFrame(all_results)
    cols = ["model", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall",
            "Specificity", "Balanced-Accuracy", "MCC", "Threshold"]
    print("\n" + results_df[cols].to_string(index=False))

    results_df.to_csv(RESULTS_DIR / "unsupervised_results.csv", index=False)

    if len(results_df) > 1:
        metrics_to_plot = ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision", "MCC"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()
        for ax, metric in zip(axes, metrics_to_plot):
            sns.barplot(data=results_df, x="model", y=metric, hue="model",
                        ax=ax, palette="Set2", legend=False)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
            ax.set_ylim(0, max(1.0, results_df[metric].max() * 1.15))
        plt.suptitle("Unsupervised Model Comparison", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "unsupervised_comparison.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"\nComparison chart → unsupervised_results/unsupervised_comparison.png")

    print(f"\nAll outputs in: {RESULTS_DIR}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsupervised wash-trading anomaly detection")
    parser.add_argument("--no-autoencoder", action="store_true",
                        help="Skip autoencoder (sklearn-only, instant)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Autoencoder epochs (default: 100)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for autoencoder (default: auto-detect)")
    args = parser.parse_args()
    main(run_autoencoder=not args.no_autoencoder,
         ae_epochs=args.epochs,
         device_str=args.device)
