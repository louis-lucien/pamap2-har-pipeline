"""
ÉTAPE Analyse exploratoire — Voir les signaux avant de modéliser.

3 figures qui justifient toute la démarche :
  1. eda_signal_traces.png : 3 s d'accéléromètre cheville pour 4 activités
     -> on VOIT que marche/course/vélo ont des signatures différentes
  2. eda_hr_by_activity.png : cardio par activité -> l'effort se lit
  3. eda_dom_freq.png : fréquence dominante par activité -> la cadence sépare
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("eda")
plt.rcParams["figure.dpi"] = 120
BLUE = "#2b6cb0"


def run_eda(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    figdir = resolve(cfg["paths"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)
    names = {int(k): v for k, v in cfg["activities"].items()}

    sig = pd.read_parquet(resolve(cfg["paths"]["interim"]))
    feats = pd.read_parquet(resolve(cfg["paths"]["features"]))

    # 1. Traces de signal brut (3 s, cheville, axe x) pour 4 activités ------
    show = [1, 4, 5, 6]   # allongé, marche, course, vélo
    fig, axes = plt.subplots(len(show), 1, figsize=(8, 6), sharex=True)
    s0 = sig[sig["subject_id"] == sig["subject_id"].iloc[0]]
    for ax, act in zip(axes, show):
        w = s0[s0["activity_id"] == act].head(300)   # 3 s à 100 Hz
        ax.plot(w["timestamp"].to_numpy() - w["timestamp"].iloc[0],
                w["ankle_acc16_x"].to_numpy(), color=BLUE, lw=0.9)
        ax.set_ylabel(names[act], rotation=0, ha="right", fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle("Accéléromètre cheville (axe x) — la signature de chaque activité")
    fig.tight_layout(); fig.savefig(figdir / "eda_signal_traces.png"); plt.close(fig)

    # 2. Cardio par activité --------------------------------------------------
    order = sorted(names.keys())
    data = [feats.loc[feats["activity_id"] == a, "hr_mean"].dropna()
            for a in order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot(data, tick_labels=[names[a] for a in order])
    ax.set_ylabel("Fréquence cardiaque (bpm)")
    ax.set_title("Le cœur raconte l'effort")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(figdir / "eda_hr_by_activity.png"); plt.close(fig)

    # 3. Fréquence dominante (cheville) par activité --------------------------
    data = [feats.loc[feats["activity_id"] == a, "ankle_dom_freq"].dropna()
            for a in order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.boxplot(data, tick_labels=[names[a] for a in order])
    ax.set_ylabel("Fréquence dominante (Hz)")
    ax.set_title("La cadence (FFT cheville) sépare marche, course et vélo")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(figdir / "eda_dom_freq.png"); plt.close(fig)

    log.info("3 figures EDA écrites dans %s", figdir)


if __name__ == "__main__":
    run_eda()
