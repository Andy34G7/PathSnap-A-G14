"""Optuna TPE Bayesian hyperparameter optimisation for the selected LightGBM classifier."""

from __future__ import annotations

import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from src.features.feature_engineering import FEATURES_CACHE_DIR
from src.models.train_and_compare import (
    MODELS_ARTIFACTS_DIR,
    MODELS_FIGURES_DIR,
    MODELS_REPORTS_DIR,
    compute_top1_and_continuity,
    load_partition_arcs,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_lgbm_cv_and_test(
    params: dict[str, Any],
    X_tr_3d: np.ndarray,
    y_tr_2d: np.ndarray,
    cand_tr: np.ndarray,
    gt_tr: np.ndarray,
    trip_idx_tr: np.ndarray,
    trip_names_tr: np.ndarray,
    offsets_tr: np.ndarray,
    arcs_tr_map: dict[str, np.ndarray],
    X_te_3d: np.ndarray,
    y_te_2d: np.ndarray,
    cand_te: np.ndarray,
    gt_te: np.ndarray,
    trip_names_te: np.ndarray,
    offsets_te: np.ndarray,
    arcs_te_map: dict[str, np.ndarray],
    label: str,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Run exact 5-fold GroupKFold CV ([::3] train stride) + full student_test evaluation."""
    n_tr_obs, k, n_feat = X_tr_3d.shape
    n_te_obs = len(gt_te)
    y_tr_flat = y_tr_2d.reshape(-1)
    y_te_flat = y_te_2d.reshape(-1)
    X_te_flat = X_te_3d.reshape(-1, n_feat)

    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros((n_tr_obs, k), dtype=np.float64)

    for tr_obs_idx, va_obs_idx in gkf.split(np.arange(n_tr_obs), groups=trip_idx_tr):
        fit_obs = tr_obs_idx[::3]
        X_fit = X_tr_3d[fit_obs].reshape(-1, n_feat)
        y_fit = y_tr_2d[fit_obs].reshape(-1)
        X_val = X_tr_3d[va_obs_idx].reshape(-1, n_feat)

        clf = LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
        clf.fit(X_fit, y_fit)
        oof_probs[va_obs_idx] = clf.predict_proba(X_val)[:, 1].reshape(-1, k)

    oof_flat = np.clip(oof_probs.reshape(-1), 1e-7, 1.0 - 1e-7)
    cv_f1 = float(f1_score(y_tr_flat, (oof_flat >= 0.5).astype(np.uint8), average="macro"))
    cv_ll = float(log_loss(y_tr_flat, oof_flat))
    cv_top1, cv_cont, _ = compute_top1_and_continuity(
        oof_probs, cand_tr, gt_tr, trip_names_tr, offsets_tr, arcs_tr_map
    )

    full_fit_obs = np.arange(0, n_tr_obs, 3)
    X_full = X_tr_3d[full_fit_obs].reshape(-1, n_feat)
    y_full = y_tr_2d[full_fit_obs].reshape(-1)

    t0 = time.perf_counter()
    final_clf = LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
    final_clf.fit(X_full, y_full)
    train_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    te_probs = final_clf.predict_proba(X_te_flat)[:, 1]
    inf_time_s = time.perf_counter() - t1

    te_probs_2d = te_probs.reshape(n_te_obs, k)
    te_probs_clip = np.clip(te_probs, 1e-7, 1.0 - 1e-7)
    te_pred = (te_probs >= 0.5).astype(np.uint8)

    te_f1 = float(f1_score(y_te_flat, te_pred, average="macro"))
    te_prec = float(precision_score(y_te_flat, te_pred, zero_division=0))
    te_rec = float(recall_score(y_te_flat, te_pred, zero_division=0))
    te_ll = float(log_loss(y_te_flat, te_probs_clip))
    te_top1, te_cont, _ = compute_top1_and_continuity(
        te_probs_2d, cand_te, gt_te, trip_names_te, offsets_te, arcs_te_map
    )

    if save_path is not None:
        joblib.dump(final_clf, save_path)

    return {
        "variant": label,
        "cv_macro_f1": round(cv_f1, 4),
        "cv_top1_acc_pct": round(cv_top1, 4),
        "cv_logloss": round(cv_ll, 4),
        "cv_route_continuity_pct": round(cv_cont, 4),
        "student_test_macro_f1": round(te_f1, 4),
        "student_test_top1_acc_pct": round(te_top1, 4),
        "student_test_route_continuity_pct": round(te_cont, 4),
        "student_test_pos_precision": round(te_prec, 4),
        "student_test_pos_recall": round(te_rec, 4),
        "student_test_logloss": round(te_ll, 4),
        "train_time_s": round(train_time_s, 3),
        "inference_time_s": round(inf_time_s, 4),
    }


def main() -> None:
    MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    tr = np.load(FEATURES_CACHE_DIR / "train_k8.npz")
    te = np.load(FEATURES_CACHE_DIR / "student_test_k8.npz")

    X_tr_3d, y_tr_2d = tr["X"], tr["y"]
    cand_tr, gt_tr, trip_idx_tr = tr["cand_arcs"], tr["gt_arcs"], tr["trip_idx"]
    trip_names_tr, offsets_tr = tr["trip_names"], tr["obs_offsets"]

    X_te_3d, y_te_2d = te["X"], te["y"]
    cand_te, gt_te = te["cand_arcs"], te["gt_arcs"]
    trip_names_te, offsets_te = te["trip_names"], te["obs_offsets"]

    n_tr_obs, k, n_feat = X_tr_3d.shape
    y_tr_flat = y_tr_2d.reshape(-1)

    arcs_tr_map = load_partition_arcs("train", trip_names_tr)
    arcs_te_map = load_partition_arcs("student_test", trip_names_te)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(n_tr_obs), groups=trip_idx_tr))

    trial_records: list[dict[str, Any]] = []
    t_opt_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        t_trial = time.perf_counter()
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        oof_probs = np.zeros(n_tr_obs * k, dtype=np.float64)
        for tr_obs_idx, va_obs_idx in splits:
            # 1-in-5 observation stride inside TPE search trials for rapid 5-fold group CV
            fit_obs = tr_obs_idx[::5]
            X_fit = X_tr_3d[fit_obs].reshape(-1, n_feat)
            y_fit = y_tr_2d[fit_obs].reshape(-1)
            X_val = X_tr_3d[va_obs_idx].reshape(-1, n_feat)

            clf = LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
            clf.fit(X_fit, y_fit)
            val_probs = clf.predict_proba(X_val)[:, 1]
            # Map obs indices to flat indices
            flat_start_idx = va_obs_idx[:, None] * k + np.arange(k)[None, :]
            oof_probs[flat_start_idx.ravel()] = val_probs

        macro_f1 = float(f1_score(y_tr_flat, (oof_probs >= 0.5).astype(np.uint8), average="macro"))
        trial_records.append(
            {
                "trial": trial.number,
                "cv_macro_f1_search": round(macro_f1, 5),
                "duration_s": round(time.perf_counter() - t_trial, 2),
                **{key: round(val, 5) if isinstance(val, float) else val for key, val in params.items()},
            }
        )
        return macro_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=25)
    total_search_s = time.perf_counter() - t_opt_start

    trials_df = pd.DataFrame(trial_records)
    trials_df["best_so_far"] = trials_df["cv_macro_f1_search"].cummax()
    trials_df.to_csv(MODELS_REPORTS_DIR / "optuna_trials.csv", index=False)

    best_params = dict(study.best_params)
    best_params["subsample_freq"] = 1
    print(f"Optuna 25 trials finished in {total_search_s:.2f}s. Best search CV Macro F1={study.best_value:.5f}")
    print(f"Best parameters: {best_params}")

    untuned_params = {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    }

    untuned_res = evaluate_lgbm_cv_and_test(
        untuned_params,
        X_tr_3d, y_tr_2d, cand_tr, gt_tr, trip_idx_tr, trip_names_tr, offsets_tr, arcs_tr_map,
        X_te_3d, y_te_2d, cand_te, gt_te, trip_names_te, offsets_te, arcs_te_map,
        label="Untuned LightGBM",
    )
    tuned_res = evaluate_lgbm_cv_and_test(
        best_params,
        X_tr_3d, y_tr_2d, cand_tr, gt_tr, trip_idx_tr, trip_names_tr, offsets_tr, arcs_tr_map,
        X_te_3d, y_te_2d, cand_te, gt_te, trip_names_te, offsets_te, arcs_te_map,
        label="Tuned LightGBM (Optuna TPE)",
        save_path=str(MODELS_ARTIFACTS_DIR / "best_lgbm_tuned.joblib"),
    )

    comp_df = pd.DataFrame([untuned_res, tuned_res])
    comp_df.to_csv(MODELS_REPORTS_DIR / "untuned_vs_tuned_comparison.csv", index=False)
    print("\n=== Untuned vs. Tuned LightGBM ===")
    print(comp_df.to_string(index=False))

    # Plot Optuna convergence and untuned vs tuned comparison
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    axes[0].plot(trials_df["trial"], trials_df["cv_macro_f1_search"], "o-", color="#64748b", alpha=0.7, label="Trial CV Macro F1")
    axes[0].plot(trials_df["trial"], trials_df["best_so_far"], color="#2563eb", linewidth=2.2, label="Best-So-Far Envelope")
    axes[0].set_xlabel("Optuna TPE Trial")
    axes[0].set_ylabel("5-Fold GroupKFold Macro F1")
    axes[0].set_title(f"Bayesian Optimisation History (25 Trials, {total_search_s:.1f}s)")
    axes[0].legend()

    metrics_labels = ["CV Macro F1", "Test Macro F1", "Test Top-1 (% / 100)", "Test Continuity (% / 100)"]
    u_vals = [
        untuned_res["cv_macro_f1"],
        untuned_res["student_test_macro_f1"],
        untuned_res["student_test_top1_acc_pct"] / 100.0,
        untuned_res["student_test_route_continuity_pct"] / 100.0,
    ]
    t_vals = [
        tuned_res["cv_macro_f1"],
        tuned_res["student_test_macro_f1"],
        tuned_res["student_test_top1_acc_pct"] / 100.0,
        tuned_res["student_test_route_continuity_pct"] / 100.0,
    ]
    x = np.arange(len(metrics_labels))
    w = 0.35
    b1 = axes[1].bar(x - w / 2, u_vals, width=w, color="#64748b", label="Untuned LightGBM")
    b2 = axes[1].bar(x + w / 2, t_vals, width=w, color="#16a34a", label="Tuned LightGBM")
    for bars in (b1, b2):
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_labels, rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylim(0.8, 1.03)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Untuned vs. Bayesian-Tuned LightGBM")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(MODELS_FIGURES_DIR / "fig5_optuna_convergence_and_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
