"""
Higgs Boson Classifier — Step 4: Inference on Real Data
========================================================
Applies the trained XGBoost model to real collision data and
generates the m4l plot showing the Higgs peak at ~125 GeV.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "higgs_xgb_model.json")

# Score threshold for "Higgs-like" event selection
# Load from training if available, otherwise fallback
SCORE_THRESHOLD_FILE = os.path.join(OUTPUT_DIR, "optimal_threshold.npy")
if os.path.exists(SCORE_THRESHOLD_FILE):
    SCORE_THRESHOLD = float(np.load(SCORE_THRESHOLD_FILE)[0])
else:
    SCORE_THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Higgs Boson Classifier -- Inference on Real Data")
    print("=" * 60)

    # ─── Load model ───
    print("\nLoading trained model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"  Model loaded from: {MODEL_PATH}")

    # ─── Load real data features ───
    print("\nLoading real data features...")
    data = np.load(os.path.join(DATA_DIR, "data_features.npz"),
                   allow_pickle=True)
    X_data_full = data["X"]
    m4l = data["m4l"]
    feature_names_all = list(data["feature_names"])

    # Load the training feature names (subset used during training)
    train_feature_names = list(np.load(
        os.path.join(OUTPUT_DIR, "train_feature_names.npy"),
        allow_pickle=True))
    
    # Select only the features used in training
    keep_idx = [feature_names_all.index(f) for f in train_feature_names]
    X_data = X_data_full[:, keep_idx]

    print(f"  Real data events: {X_data.shape[0]:,}")
    print(f"  All features: {len(feature_names_all)}")
    print(f"  Training features: {len(train_feature_names)}")
    print(f"  Excluded: {set(feature_names_all) - set(train_feature_names)}")

    # Handle NaN/Inf
    valid = ~np.any(np.isnan(X_data) | np.isinf(X_data), axis=1)
    X_data = X_data[valid]
    m4l = m4l[valid]
    print(f"  After NaN removal: {X_data.shape[0]:,} events")

    # ─── Run inference ───
    print("\nRunning inference...")
    scores = model.predict_proba(X_data)[:, 1]

    # Classify
    is_higgs = scores >= SCORE_THRESHOLD
    n_higgs = np.sum(is_higgs)
    print(f"  Events classified as Higgs-like (score >= {SCORE_THRESHOLD}): "
          f"{n_higgs:,} / {X_data.shape[0]:,}")

    # ─── Score distribution ───
    print("\nScore distribution:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        n = np.sum(scores >= threshold)
        print(f"  score >= {threshold}: {n:,} events")

    # ─── Plots ───
    print("\nGenerating plots...")

    # 1. XGBoost score distribution on real data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(scores, bins=50, range=(0, 1), histtype='stepfilled',
            alpha=0.7, color='steelblue', edgecolor='navy')
    ax.axvline(SCORE_THRESHOLD, color='red', ls='--', lw=2,
               label=f'Threshold = {SCORE_THRESHOLD}')
    ax.set_xlabel('XGBoost Signal Score', fontsize=12)
    ax.set_ylabel('Events', fontsize=12)
    ax.set_title('XGBoost Score Distribution on Real Collision Data',
                 fontsize=14)
    ax.legend(fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "score_distribution_data.png"),
                dpi=150)
    plt.close(fig)
    print(f"  Saved score_distribution_data.png")

    # 2. THE KEY PLOT: m4l distribution before and after XGBoost cut
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bins_m4l = np.linspace(80, 250, 51)

    # Left: All events
    axes[0].hist(m4l, bins=bins_m4l, histtype='stepfilled',
                 alpha=0.7, color='gray', edgecolor='black',
                 label=f'All events ({len(m4l):,})')
    axes[0].axvline(125, color='red', ls='--', lw=2,
                    label='125 GeV (Higgs)')
    axes[0].set_xlabel(r'$m_{4\ell}$ [GeV]', fontsize=12)
    axes[0].set_ylabel('Events / 3.4 GeV', fontsize=12)
    axes[0].set_title(r'$m_{4\ell}$ — All Events (No ML Cut)', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Right: After XGBoost signal cut
    axes[1].hist(m4l[is_higgs], bins=bins_m4l, histtype='stepfilled',
                 alpha=0.7, color='royalblue', edgecolor='navy',
                 label=f'Higgs-like events ({n_higgs:,})')
    axes[1].hist(m4l[~is_higgs], bins=bins_m4l, histtype='step',
                 lw=1.5, color='gray', alpha=0.5,
                 label=f'Rejected events ({len(m4l) - n_higgs:,})')
    axes[1].axvline(125, color='red', ls='--', lw=2,
                    label='125 GeV (Higgs)')
    axes[1].set_xlabel(r'$m_{4\ell}$ [GeV]', fontsize=12)
    axes[1].set_ylabel('Events / 3.4 GeV', fontsize=12)
    axes[1].set_title(
        fr'$m_{{4\ell}}$ — After XGBoost Cut (score $\geq$ {SCORE_THRESHOLD})',
        fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('ATLAS Open Data — Higgs Boson Search in '
                 r'$H \to ZZ^* \to 4\ell$',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "m4l_higgs_search.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved m4l_higgs_search.png")

    # 3. Zoomed m4l around Higgs peak (110-140 GeV)
    fig, ax = plt.subplots(figsize=(8, 6))
    bins_zoom = np.linspace(110, 140, 31)

    ax.hist(m4l[is_higgs], bins=bins_zoom, histtype='stepfilled',
            alpha=0.7, color='royalblue', edgecolor='navy',
            label=f'Higgs-like (score >= {SCORE_THRESHOLD})')
    ax.hist(m4l[~is_higgs], bins=bins_zoom, histtype='step',
            lw=1.5, color='gray', alpha=0.5, label='Rejected')
    ax.axvline(125, color='red', ls='--', lw=2, label='125 GeV')

    ax.set_xlabel(r'$m_{4\ell}$ [GeV]', fontsize=12)
    ax.set_ylabel('Events / 1 GeV', fontsize=12)
    ax.set_title(r'Higgs Peak Region: $m_{4\ell}$ [110-140 GeV]', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "m4l_higgs_peak_zoom.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved m4l_higgs_peak_zoom.png")

    # 4. Score vs m4l 2D scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    # Sample if too many events
    n_plot = min(50000, len(scores))
    idx = np.random.choice(len(scores), n_plot, replace=False)
    sc = ax.scatter(m4l[idx], scores[idx], c=scores[idx], cmap='RdYlGn',
                    s=1, alpha=0.3, vmin=0, vmax=1)
    ax.axvline(125, color='red', ls='--', lw=2, alpha=0.7, label='125 GeV')
    ax.axhline(SCORE_THRESHOLD, color='black', ls=':', lw=1.5,
               label=f'Threshold = {SCORE_THRESHOLD}')
    ax.set_xlabel(r'$m_{4\ell}$ [GeV]', fontsize=12)
    ax.set_ylabel('XGBoost Signal Score', fontsize=12)
    ax.set_title('Signal Score vs Four-Lepton Invariant Mass', fontsize=14)
    ax.set_xlim(80, 250)
    ax.legend(fontsize=11)
    plt.colorbar(sc, ax=ax, label='Score')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "score_vs_m4l.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved score_vs_m4l.png")

    # ─── Summary Statistics ───
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Events in Higgs mass window (120-130 GeV) after cut
    higgs_window = (m4l > 120) & (m4l < 130)
    higgs_selected = is_higgs & higgs_window

    print(f"  Total real data events:     {len(m4l):>10,}")
    print(f"  Events passing ML cut:      {n_higgs:>10,}")
    print(f"  Events in 120-130 GeV:      {np.sum(higgs_window):>10,}")
    print(f"  Selected in 120-130 GeV:    {np.sum(higgs_selected):>10,}")
    print(f"  Efficiency in Higgs window: "
          f"{np.sum(higgs_selected)/max(np.sum(higgs_window), 1)*100:.1f}%")

    print(f"\n  Output plots in: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nDone! Check the m4l plots to see the Higgs peak at ~125 GeV.")


if __name__ == "__main__":
    main()
