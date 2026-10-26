"""
ÉTAPE Évaluation — Performance sur des SUJETS jamais vus.

Métriques multi-classes : accuracy, F1-macro (traite chaque activité à
égalité), rapport par classe. Figures : matrice de confusion 8x8,
importance des variables, et la figure-clé « fuite vs honnête ».
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

from src.config import get_logger, load_config, resolve

log = get_logger("evaluate")
plt.rcParams["figure.dpi"] = 120


def evaluate(model, X_te, y_te, best_params, cv_f1, extras,
             cfg: dict | None = None):
    cfg = cfg or load_config()
    figdir = resolve(cfg["paths"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)
    names = {int(k): v for k, v in cfg["activities"].items()}
    labels = sorted(y_te.unique())
    display = [names[i] for i in labels]

    pred = model.predict(X_te)
    metrics = {
        "f1_macro_cv_grouped": round(float(cv_f1), 4),
        "f1_macro_test_subject_wise": round(float(
            f1_score(y_te, pred, average="macro")), 4),
        "accuracy_test": round(float(accuracy_score(y_te, pred)), 4),
        "naive_split_f1_LEAKY": extras["naive_split_f1"],
        "leakage_gap": round(extras["naive_split_f1"] -
                             float(f1_score(y_te, pred, average="macro")), 4),
        "baseline_logistic_regression": extras["baseline"],
        "champion_deployed": extras["champion"],
        "challenger_cv": extras["challenger_rf_or_xgb_cv"],
        "best_params": best_params,
        "train_subjects": extras["train_subjects"],
        "test_subjects": extras["test_subjects"],
        "n_test_windows": int(len(y_te)),
    }
    with open(resolve(cfg["paths"]["metrics"]), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("F1-macro (test, sujets jamais vus) : %.3f",
             metrics["f1_macro_test_subject_wise"])
    log.info("Accuracy (test) : %.3f", metrics["accuracy_test"])
    log.info("Écart fuite (naïf %.3f - honnête %.3f) : %+.3f",
             metrics["naive_split_f1_LEAKY"],
             metrics["f1_macro_test_subject_wise"], metrics["leakage_gap"])
    print("\n" + classification_report(y_te, pred, labels=labels,
                                       target_names=display))

    # --- Matrice de confusion 8x8 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, labels=labels, display_labels=display,
        cmap="Blues", ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("Matrice de confusion — sujets jamais vus")
    fig.tight_layout(); fig.savefig(figdir / "confusion_matrix.png"); plt.close(fig)

    # --- Figure-clé : split naïf vs split par sujet ---
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    vals = [metrics["naive_split_f1_LEAKY"],
            metrics["f1_macro_test_subject_wise"]]
    bars = ax.bar(["Split naïf\n(fenêtres au hasard)\n= FUITE",
                   "Split par sujet\n(honnête)"],
                  vals, color=["#c53030", "#2b6cb0"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.12); ax.set_ylabel("F1-macro")
    ax.set_title("Le score « facile » est une illusion")
    fig.tight_layout(); fig.savefig(figdir / "leakage_comparison.png"); plt.close(fig)

    # --- Importance des variables (top 15) — RF/XGB ou coefficients LogReg ---
    try:
        est = model.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
            title = "Top 15 des caractéristiques les plus discriminantes"
        else:  # LogisticRegression : moyenne des |coefficients| sur les classes
            vals = np.abs(est.coef_).mean(axis=0)
            title = "Top 15 des caractéristiques (|coefficients| moyens)"
        imp = pd.Series(vals, index=X_te.columns).sort_values(
            ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(ax=ax, color="#2b6cb0")
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(figdir / "feature_importance.png")
        plt.close(fig)
    except Exception as e:  # noqa
        log.warning("Importance non disponible : %s", e)

    log.info("Figures écrites dans %s", figdir)
    return metrics


if __name__ == "__main__":
    from src.train import train
    cfg = load_config()
    model, X_te, y_te, bp, cv_f1, extras = train(cfg)
    evaluate(model, X_te, y_te, bp, cv_f1, extras, cfg)
