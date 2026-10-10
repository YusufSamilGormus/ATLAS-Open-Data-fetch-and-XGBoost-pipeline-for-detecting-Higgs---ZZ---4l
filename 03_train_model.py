"""
Higgs Boson Classifier — Step 3: XGBoost Model Training
========================================================
Trains an XGBoost classifier to distinguish Higgs signal from background
using physics-informed features computed in Step 2.

Outputs:
  - Trained model (model.json)
  - ROC curve plot
  - Feature importance plot
  - Overtraining check (train vs test distributions)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "data"
OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "higgs_xgb_model.json")

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.5,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "base_score": 0.5,
    "scale_pos_weight": 1.0,  # Classes already balanced via downsampling
    "random_state": 42,
    "n_jobs": -1,
}

# Train/test split
TEST_SIZE = 0.25
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Higgs Boson Classifier -- XGBoost Training")
    print("=" * 60)

    # ─── Load features ───
    print("\nLoading engineered features...")
    train_data = np.load(os.path.join(DATA_DIR, "train_features.npz"),
                         allow_pickle=True)
    X = train_data["X"]
    y = train_data["y"]
    weights = train_data["weights"]
    feature_names = list(train_data["feature_names"])

    print(f"  Total events: {X.shape[0]:,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Signal: {np.sum(y == 1):,}  |  Background: {np.sum(y == 0):,}")

    # ─── CRITICAL: Exclude m4l from training features ───
    # m4l IS the variable we plot to find the Higgs peak.
    # If we train on it, the model just learns "125 GeV = signal" which is
    # circular reasoning. We want the model to learn from OTHER kinematics
    # (isolation, pT, angular variables, mZ1, mZ2) and then see if the
    # selected events naturally cluster at 125 GeV in m4l.
    # We keep mZ1, mZ2 since they provide physics info about Z pairing quality.
    exclude_features = {"m4l"}
    keep_mask = [i for i, f in enumerate(feature_names)
                 if f not in exclude_features]
    excluded = [f for f in feature_names if f in exclude_features]
    feature_names_train = [feature_names[i] for i in keep_mask]
    X = X[:, keep_mask]
    print(f"\n  Excluded from training: {excluded}")
    print(f"  Training features ({len(feature_names_train)}): {feature_names_train}")

    # Save the training feature names for inference
    np.save(os.path.join(OUTPUT_DIR, "train_feature_names.npy"),
            np.array(feature_names_train))

    # Handle NaN/Inf (XGBoost handles NaN natively, but let's be safe)
    nan_mask = np.any(np.isnan(X) | np.isinf(X), axis=1)
    if np.sum(nan_mask) > 0:
        print(f"  Removing {np.sum(nan_mask):,} events with NaN/Inf values")
        X = X[~nan_mask]
        y = y[~nan_mask]
        weights = weights[~nan_mask]

    # ─── Handle weights ───
    # MC weights (mcWeight * scalefactors) normalize MC to luminosity.
    # For CLASSIFICATION training, we use uniform weights since we just
    # want to learn kinematic differences between signal and background.
    # The MC weights will be used later for physics-level normalization
    # in the final m4l plot, not here.
    print("  Using uniform weights for classification training")

    # ─── Balance classes via downsampling ───
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    print(f"\n  Before balancing: Signal={n_pos:,}, Background={n_neg:,}")

    if n_pos > n_neg:
        # Downsample signal to match background
        rng = np.random.RandomState(RANDOM_STATE)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        sampled_pos = rng.choice(pos_idx, size=n_neg, replace=False)
        keep_idx = np.sort(np.concatenate([sampled_pos, neg_idx]))
        X = X[keep_idx]
        y = y[keep_idx]
        print(f"  After balancing:  Signal={np.sum(y==1):,}, "
              f"Background={np.sum(y==0):,}")
    
    XGB_PARAMS["scale_pos_weight"] = 1.0

    # ─── Train/test split ───
    print(f"\nSplitting data ({1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y)

    print(f"  Train: {X_train.shape[0]:,} events "
          f"(sig: {np.sum(y_train == 1):,}, bkg: {np.sum(y_train == 0):,})")
    print(f"  Test:  {X_test.shape[0]:,} events "
          f"(sig: {np.sum(y_test == 1):,}, bkg: {np.sum(y_test == 0):,})")

    # ─── Train XGBoost ───
    print("\nTraining XGBoost...")
    print(f"  Parameters: max_depth={XGB_PARAMS['max_depth']}, "
          f"lr={XGB_PARAMS['learning_rate']}, "
          f"n_estimators={XGB_PARAMS['n_estimators']}")

    model = xgb.XGBClassifier(**XGB_PARAMS)

    # Fit with eval set (no sample weights — uniform weighting)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    # ─── Evaluate ───
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Predictions
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # AUC
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    print(f"\n  AUC (train): {auc_train:.4f}")
    print(f"  AUC (test):  {auc_test:.4f}")
    print(f"  Overtraining check: {'OK' if abs(auc_train - auc_test) < 0.02 else 'WARNING - possible overtraining'}")

    # Find optimal threshold (maximize Youden's J = TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    # Classification report with optimal threshold
    y_pred_optimal = (y_pred_proba_test >= optimal_threshold).astype(int)
    print(f"\n  Classification Report (test set, threshold={optimal_threshold:.3f}):")
    print(classification_report(y_test, y_pred_optimal,
                                 target_names=["Background", "Signal"],
                                 zero_division=0))

    # Save optimal threshold for inference
    np.save(os.path.join(OUTPUT_DIR, "optimal_threshold.npy"),
            np.array([optimal_threshold]))

    # ─── Save model ───
    model.save_model(MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")

    # ─── Plots ───
    print("\nGenerating plots...")

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'XGBoost (AUC = {auc_test:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.plot(fpr[best_idx], tpr[best_idx], 'ro', ms=10,
            label=f'Optimal (t={optimal_threshold:.3f})')
    ax.set_xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
    ax.set_title('ROC Curve: Higgs Signal vs Background', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved roc_curve.png")

    # 2. Feature Importance (use feature_names_train, not feature_names)
    importance = model.feature_importances_
    idx_sorted = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feature_names_train)),
            importance[idx_sorted], color='steelblue')
    ax.set_yticks(range(len(feature_names_train)))
    ax.set_yticklabels([feature_names_train[i] for i in idx_sorted])
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost Feature Importance', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved feature_importance.png")

    # 3. Overtraining Check — BDT score distributions
    fig, ax = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1, 51)

    ax.hist(y_pred_proba_train[y_train == 1], bins=bins, density=True,
            histtype='step', lw=2, color='blue', label='Signal (train)')
    ax.hist(y_pred_proba_train[y_train == 0], bins=bins, density=True,
            histtype='step', lw=2, color='red', label='Background (train)')
    ax.hist(y_pred_proba_test[y_test == 1], bins=bins, density=True,
            histtype='stepfilled', alpha=0.3, color='blue',
            label='Signal (test)')
    ax.hist(y_pred_proba_test[y_test == 0], bins=bins, density=True,
            histtype='stepfilled', alpha=0.3, color='red',
            label='Background (test)')
    ax.axvline(optimal_threshold, color='green', ls='--', lw=2,
               label=f'Threshold = {optimal_threshold:.3f}')

    ax.set_xlabel('XGBoost Score', fontsize=12)
    ax.set_ylabel('Normalized Events', fontsize=12)
    ax.set_title('Overtraining Check', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "overtraining_check.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved overtraining_check.png")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:  {MODEL_PATH}")
    print(f"  AUC:    {auc_test:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Plots:  {OUTPUT_DIR}/")
    print(f"\n  Next step: Run 04_inference.py")


if __name__ == "__main__":
    main()
