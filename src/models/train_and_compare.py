"""Train and compare 6 distinct ML model families with 5-fold trip-grouped CV and student-test evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.eda.inventory_and_quality import BASE_DIR, load_trip_raw
from src.features.feature_engineering import FEATURES_CACHE_DIR

MODELS_REPORTS_DIR = BASE_DIR / "reports" / "models"
MODELS_FIGURES_DIR = MODELS_REPORTS_DIR / "figures"
MODELS_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "models"


def load_partition_arcs(partition: str, trip_names: np.ndarray) -> dict[str, np.ndarray]:
    """Load only the .arcs edge tables for computing route continuity."""
    arcs_map: dict[str, np.ndarray] = {}
    for tid in trip_names:
        data = load_trip_raw(BASE_DIR / partition / str(tid))
        arcs_map[str(tid)] = data["arcs"]
    return arcs_map


def compute_top1_and_continuity(
    probs_2d: np.ndarray,
    cand_arcs: np.ndarray,
    gt_arcs: np.ndarray,
    trip_names: np.ndarray,
    obs_offsets: np.ndarray,
    arcs_map: dict[str, np.ndarray],
) -> tuple[float, float, np.ndarray]:
    """Compute observation-level Top-1 Accuracy (%) and directed Route Continuity (%)."""
    n_obs = len(gt_arcs)
    top_k = np.argmax(probs_2d, axis=1)
    pred_arcs = cand_arcs[np.arange(n_obs), top_k]
    top1_acc = 100.0 * float(np.mean(pred_arcs == gt_arcs))

    total_transitions = 0
    connected_transitions = 0
    for idx, tid in enumerate(trip_names):
        s, e = int(obs_offsets[idx]), int(obs_offsets[idx + 1])
        seq = pred_arcs[s:e]
        if len(seq) < 2:
            continue
        arcs = arcs_map[str(tid)]
        same = seq[1:] == seq[:-1]
        adj = arcs[seq[:-1], 1] == arcs[seq[1:], 0]
        connected_transitions += int(np.sum(same | adj))
        total_transitions += len(seq) - 1

    continuity = 100.0 * connected_transitions / total_transitions if total_transitions else 0.0
    return top1_acc, continuity, pred_arcs


def get_model_registry(seed: int = 42) -> list[dict[str, Any]]:
    """Return 6 models spanning 6 distinct algorithm families."""
    return [
        {
            "name": "Logistic Regression",
            "family": "Linear",
            "config_summary": "StandardScaler + L2 LogisticRegression(C=1.0, max_iter=500)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=seed)),
                ]
            ),
        },
        {
            "name": "Gaussian Naive Bayes",
            "family": "Probabilistic",
            "config_summary": "StandardScaler + GaussianNB(var_smoothing=1e-6)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", GaussianNB(var_smoothing=1e-6)),
                ]
            ),
        },
        {
            "name": "Decision Tree",
            "family": "Tree-based",
            "config_summary": "DecisionTreeClassifier(max_depth=12, min_samples_leaf=50)",
            "estimator": DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed),
        },
        {
            "name": "Random Forest",
            "family": "Ensemble Bagging",
            "config_summary": "RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=20)",
            "estimator": RandomForestClassifier(
                n_estimators=80, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=seed
            ),
        },
        {
            "name": "LightGBM",
            "family": "Gradient Boosting",
            "config_summary": "LGBMClassifier(n_estimators=200, lr=0.08, num_leaves=31, min_child=50, subsample_freq=1)",
            "estimator": LGBMClassifier(
                n_estimators=200,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
            ),
        },
        {
            "name": "MLP Neural Network",
            "family": "Neural Networks",
            "config_summary": "StandardScaler + MLP(hidden=(64,32), early_stopping=True, batch=2048)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            max_iter=35,
                            batch_size=2048,
                            tol=1e-3,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=4,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
    ]


def evaluate_models() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Run 5-fold trip-grouped CV on train and evaluate all 6 models on student_test."""
    tr = np.load(FEATURES_CACHE_DIR / "train_k8.npz")
    te = np.load(FEATURES_CACHE_DIR / "student_test_k8.npz")

    X_tr_3d, y_tr_2d = tr["X"], tr["y"]
    cand_tr, gt_tr = tr["cand_arcs"], tr["gt_arcs"]
    trip_idx_tr = tr["trip_idx"]
    trip_names_tr, offsets_tr = tr["trip_names"], tr["obs_offsets"]

    X_te_3d, y_te_2d = te["X"], te["y"]
    cand_te, gt_te = te["cand_arcs"], te["gt_arcs"]
    trip_names_te, offsets_te = te["trip_names"], te["obs_offsets"]

    n_tr_obs, k, n_feat = X_tr_3d.shape
    n_te_obs = len(gt_te)

    X_te_flat = X_te_3d.reshape(-1, n_feat)
    y_te_flat = y_te_2d.reshape(-1)
    y_tr_flat = y_tr_2d.reshape(-1)

    arcs_tr_map = load_partition_arcs("train", trip_names_tr)
    arcs_te_map = load_partition_arcs("student_test", trip_names_te)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(n_tr_obs), groups=trip_idx_tr))

    registry = get_model_registry(seed=42)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cms: dict[str, np.ndarray] = {}
    histories: dict[str, Any] = {}

    for spec in registry:
        name = spec["name"]
        oof_probs = np.zeros((n_tr_obs, k), dtype=np.float64)
        fold_f1s = []
        fold_top1s = []

        for fold_id, (tr_obs_idx, va_obs_idx) in enumerate(splits):
            # Stride training observations by 3 inside each CV fold for balanced speed
            fit_obs = tr_obs_idx[::3]
            X_fit = X_tr_3d[fit_obs].reshape(-1, n_feat)
            y_fit = y_tr_2d[fit_obs].reshape(-1)
            X_val = X_tr_3d[va_obs_idx].reshape(-1, n_feat)
            y_val = y_tr_2d[va_obs_idx].reshape(-1)

            est = clone(spec["estimator"])
            if name == "LightGBM" and fold_id == 0:
                est.fit(
                    X_fit,
                    y_fit,
                    eval_set=[(X_fit, y_fit), (X_val, y_val)],
                    eval_metric="binary_logloss",
                )
                histories["LightGBM"] = est.evals_result_
            else:
                est.fit(X_fit, y_fit)

            val_probs = est.predict_proba(X_val)[:, 1]
            oof_probs[va_obs_idx] = val_probs.reshape(-1, k)

            val_pred = (val_probs >= 0.5).astype(np.uint8)
            f_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
            f_top1 = 100.0 * float(
                np.mean(
                    cand_tr[va_obs_idx, np.argmax(oof_probs[va_obs_idx], axis=1)] == gt_tr[va_obs_idx]
                )
            )
            fold_f1s.append(f_macro_f1)
            fold_top1s.append(f_top1)
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold_id,
                    "val_macro_f1": round(f_macro_f1, 4),
                    "val_top1_acc_pct": round(f_top1, 4),
                }
            )

        oof_flat = np.clip(oof_probs.reshape(-1), 1e-7, 1.0 - 1e-7)
        oof_pred = (oof_flat >= 0.5).astype(np.uint8)
        cv_macro_f1 = float(f1_score(y_tr_flat, oof_pred, average="macro"))
        cv_logloss = float(log_loss(y_tr_flat, oof_flat))
        cv_top1, cv_cont, _ = compute_top1_and_continuity(
            oof_probs, cand_tr, gt_tr, trip_names_tr, offsets_tr, arcs_tr_map
        )

        # Refit on strided full training set ([::3]) and evaluate on 100% of student_test
        full_fit_obs = np.arange(0, n_tr_obs, 3)
        X_full_fit = X_tr_3d[full_fit_obs].reshape(-1, n_feat)
        y_full_fit = y_tr_2d[full_fit_obs].reshape(-1)
        final_est = clone(spec["estimator"])

        t0 = time.perf_counter()
        final_est.fit(X_full_fit, y_full_fit)
        train_time_s = time.perf_counter() - t0

        if name == "MLP Neural Network":
            mlp_clf = final_est.named_steps["clf"]
            histories["MLP"] = {
                "loss_curve": list(mlp_clf.loss_curve_),
                "val_scores": list(mlp_clf.validation_scores_) if hasattr(mlp_clf, "validation_scores_") else [],
            }

        # Measure training-fit Macro F1 on the full fit subset to check overfitting
        tr_fit_probs = final_est.predict_proba(X_full_fit)[:, 1]
        tr_fit_f1 = float(f1_score(y_full_fit, (tr_fit_probs >= 0.5).astype(np.uint8), average="macro"))

        t1 = time.perf_counter()
        te_probs_flat = final_est.predict_proba(X_te_flat)[:, 1]
        inf_time_s = time.perf_counter() - t1

        te_probs_2d = te_probs_flat.reshape(n_te_obs, k)
        te_probs_clipped = np.clip(te_probs_flat, 1e-7, 1.0 - 1e-7)
        te_pred = (te_probs_flat >= 0.5).astype(np.uint8)

        te_macro_f1 = float(f1_score(y_te_flat, te_pred, average="macro"))
        te_pos_prec = float(precision_score(y_te_flat, te_pred, zero_division=0))
        te_pos_rec = float(recall_score(y_te_flat, te_pred, zero_division=0))
        te_pos_f1 = float(f1_score(y_te_flat, te_pred, zero_division=0))
        te_logloss = float(log_loss(y_te_flat, te_probs_clipped))
        te_top1, te_cont, _ = compute_top1_and_continuity(
            te_probs_2d, cand_te, gt_te, trip_names_te, offsets_te, arcs_te_map
        )

        cms[name] = confusion_matrix(y_te_flat, te_pred)
        if name == "LightGBM":
            joblib.dump(final_est, MODELS_ARTIFACTS_DIR / "untuned_lgbm.joblib")

        summary_rows.append(
            {
                "model": name,
                "family": spec["family"],
                "config_summary": spec["config_summary"],
                "train_fit_macro_f1": round(tr_fit_f1, 4),
                "cv_macro_f1": round(cv_macro_f1, 4),
                "cv_macro_f1_std": round(float(np.std(fold_f1s)), 4),
                "cv_top1_acc_pct": round(cv_top1, 4),
                "cv_route_continuity_pct": round(cv_cont, 4),
                "cv_logloss": round(cv_logloss, 4),
                "student_test_macro_f1": round(te_macro_f1, 4),
                "student_test_top1_acc_pct": round(te_top1, 4),
                "student_test_route_continuity_pct": round(te_cont, 4),
                "student_test_pos_precision": round(te_pos_prec, 4),
                "student_test_pos_recall": round(te_pos_rec, 4),
                "student_test_pos_f1": round(te_pos_f1, 4),
                "student_test_logloss": round(te_logloss, 4),
                "train_time_s": round(train_time_s, 3),
                "inference_time_s": round(inf_time_s, 4),
                "inference_ms_per_trip": round(1000.0 * inf_time_s / len(trip_names_te), 2),
            }
        )
        print(
            f"[{name:20s}] CV F1={cv_macro_f1:.4f} | Test F1={te_macro_f1:.4f} | "
            f"Test Top1={te_top1:.2f}% | Continuity={te_cont:.2f}% | LogLoss={te_logloss:.4f} | "
            f"Train={train_time_s:.2f}s | Inf={inf_time_s:.3f}s"
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), cms, histories


def plot_comparisons(df: pd.DataFrame, cms: dict[str, np.ndarray], histories: dict[str, Any]) -> None:
    """Generate the 4 required model comparison and training-history plots."""
    sns.set_theme(style="whitegrid")
    models = df["model"].tolist()
    x = np.arange(len(models))
    w = 0.36

    # Fig 1: CV Macro F1 & Student-Test Macro F1 + Top-1 Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    b1 = axes[0].bar(x - w / 2, df["cv_macro_f1"], width=w, color="#2563eb", label="5-Fold CV Macro F1")
    b2 = axes[0].bar(x + w / 2, df["student_test_macro_f1"], width=w, color="#ea580c", label="Student-Test Macro F1")
    for bars in (b1, b2):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel("Macro F1-score")
    axes[0].set_title("Cross-Validation vs. Student-Test Macro F1")
    axes[0].legend()

    b3 = axes[1].bar(x, df["student_test_top1_acc_pct"], width=0.52, color="#16a34a")
    for bar in b3:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylim(60, 102)
    axes[1].set_ylabel("Observation-Level Top-1 Accuracy (%)")
    axes[1].set_title("Student-Test Top-1 Candidate Accuracy")

    plt.tight_layout()
    fig.savefig(MODELS_FIGURES_DIR / "fig1_cv_and_test_model_comparison.png", dpi=180)
    plt.close(fig)

    # Fig 2: Log Loss and Route Continuity comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    b_ll = axes[0].bar(x, df["student_test_logloss"], width=0.52, color="#7c3aed")
    for bar in b_ll:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Student-Test Log Loss (lower is better)")
    axes[0].set_title("Probability Calibration (Log Loss on Student-Test)")

    b_ct = axes[1].bar(x, df["student_test_route_continuity_pct"], width=0.52, color="#0f766e")
    for bar in b_ct:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylim(75, 102)
    axes[1].set_ylabel("Classifier-Only Route Continuity (%)")
    axes[1].set_title("Student-Test Directed Route Continuity (Pre-HMM)")

    plt.tight_layout()
    fig.savefig(MODELS_FIGURES_DIR / "fig2_logloss_and_continuity_comparison.png", dpi=180)
    plt.close(fig)

    # Fig 3: Confusion Matrices for all 6 models
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
    for ax, name in zip(axes.ravel(), models):
        cm = cms[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
            ax=ax,
        )
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference Label")
    plt.tight_layout()
    fig.savefig(MODELS_FIGURES_DIR / "fig3_confusion_matrices_6models.png", dpi=180)
    plt.close(fig)

    # Fig 4: Real epoch/iteration training histories ONLY for LightGBM & MLP
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    if "LightGBM" in histories:
        lgb_res = histories["LightGBM"]
        tr_loss = lgb_res["training"]["binary_logloss"]
        va_loss = lgb_res["valid_1"]["binary_logloss"]
        rounds = np.arange(1, len(tr_loss) + 1)
        axes[0].plot(rounds, tr_loss, color="#2563eb", linewidth=2, label="Fold 0 Train LogLoss")
        axes[0].plot(rounds, va_loss, color="#ea580c", linewidth=2, label="Fold 0 Validation LogLoss")
        axes[0].set_xlabel("Boosting Iteration")
        axes[0].set_ylabel("Binary Log Loss")
        axes[0].set_title("LightGBM Iteration-Wise Training vs. Validation Loss")
        axes[0].legend()

    if "MLP" in histories:
        mlp_h = histories["MLP"]
        epochs = np.arange(1, len(mlp_h["loss_curve"]) + 1)
        axes[1].plot(epochs, mlp_h["loss_curve"], color="#2563eb", linewidth=2, marker="o", markersize=4, label="Training LogLoss")
        axes[1].set_xlabel("Training Epoch")
        axes[1].set_ylabel("Training Log Loss", color="#2563eb")
        if mlp_h["val_scores"]:
            ax2 = axes[1].twinx()
            ax2.plot(
                epochs,
                [100.0 * s for s in mlp_h["val_scores"]],
                color="#16a34a",
                linewidth=2,
                linestyle="--",
                marker="^",
                markersize=4,
                label="Internal Val Accuracy (%)",
            )
            ax2.set_ylabel("Internal Validation Accuracy (%)", color="#16a34a")
            ax2.legend(loc="center right")
        axes[1].set_title("MLP Neural Network Epoch Training Curve")
        axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(MODELS_FIGURES_DIR / "fig4_epoch_training_validation_loss.png", dpi=180)
    plt.close(fig)


def main() -> None:
    MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, fold_df, cms, histories = evaluate_models()
    summary_df.to_csv(MODELS_REPORTS_DIR / "model_comparison.csv", index=False)
    fold_df.to_csv(MODELS_REPORTS_DIR / "cv_fold_metrics.csv", index=False)
    plot_comparisons(summary_df, cms, histories)
    print("\nSaved model_comparison.csv, cv_fold_metrics.csv, untuned_lgbm.joblib, and fig1-fig4.")


if __name__ == "__main__":
    main()
