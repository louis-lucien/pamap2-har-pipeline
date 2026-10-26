"""
Générateur synthétique fidèle au format PAMAP2 (UCI ML Repository).

Reproduit EXACTEMENT le format officiel :
  - 1 fichier .dat par sujet (subject101.dat ... subject109.dat)
  - valeurs séparées par des espaces, NaN pour les manquants
  - 54 colonnes : timestamp, activityID, heart_rate, puis 3 IMU
    (main, poitrine, cheville) x 17 colonnes chacune :
    temp, acc16 x/y/z, acc6 x/y/z, gyro x/y/z, mag x/y/z, orientation x4

Et les DÉFAUTS réels que le pipeline doit gérer :
  - cardio échantillonné à ~9 Hz -> NaN entre deux mesures
  - pertes sans fil : NaN aléatoires et par rafales sur les IMU
  - colonnes d'orientation invalides (doc officielle) -> remplies de 0
  - périodes transitoires entre activités (activityID = 0)

La physique simulée par activité (gravité selon la posture, composantes
périodiques de la marche/course/vélo, cardio dépendant de l'effort) donne
un signal APPRENABLE mais réaliste, avec variabilité inter-sujets.

>>> Vrai jeu : le pipeline le télécharge depuis l'UCI (config data_source),
    ou déposer les .dat dans data/raw/Protocol. Le code ne change pas.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("generate_data")
RNG = np.random.default_rng(42)

FS = 100                    # Hz
# Durée par activité et par sujet. Surchargeable via la variable d'environnement
# PAMAP2_SECONDS (utile pour un run allégé sur une machine à faible espace disque).
SECONDS_PER_ACTIVITY = int(os.environ.get("PAMAP2_SECONDS", "45"))
TRANSIENT_SECONDS = 4       # transition (activityID = 0) entre activités

# (freq_pas Hz, amplitude cheville, amplitude main, amplitude poitrine, HR base)
ACTIVITY_PROFILES = {
    1:  dict(f=0.0, a_ankle=0.05, a_hand=0.05, a_chest=0.03, hr=65,  posture="lying"),
    2:  dict(f=0.0, a_ankle=0.06, a_hand=0.15, a_chest=0.05, hr=72,  posture="sit"),
    3:  dict(f=0.0, a_ankle=0.08, a_hand=0.20, a_chest=0.08, hr=78,  posture="stand"),
    4:  dict(f=1.9, a_ankle=3.5,  a_hand=1.2,  a_chest=0.8,  hr=100, posture="stand"),
    5:  dict(f=2.9, a_ankle=8.0,  a_hand=3.0,  a_chest=2.5,  hr=155, posture="stand"),
    6:  dict(f=1.3, a_ankle=4.0,  a_hand=0.5,  a_chest=0.4,  hr=125, posture="sit"),
    12: dict(f=1.5, a_ankle=4.2,  a_hand=1.5,  a_chest=1.2,  hr=120, posture="stand"),
    13: dict(f=1.7, a_ankle=3.8,  a_hand=1.3,  a_chest=1.0,  hr=108, posture="stand"),
}

GRAVITY = {  # vecteur gravité (x, y, z) par posture et par IMU
    "lying": {"hand": (9.5, 1.0, 1.5), "chest": (0.5, 1.0, 9.6), "ankle": (9.4, 1.2, 1.8)},
    "sit":   {"hand": (2.0, 9.2, 2.0), "chest": (1.5, 9.4, 2.0), "ankle": (7.0, 5.5, 2.5)},
    "stand": {"hand": (1.0, 9.6, 1.0), "chest": (0.8, 9.7, 1.0), "ankle": (0.8, 9.7, 1.2)},
}


def _imu_block(n: int, activity: dict, loc: str, subj_amp: float,
               subj_freq: float) -> np.ndarray:
    """Génère les 17 colonnes d'un IMU pour n échantillons."""
    t = np.arange(n) / FS
    g = np.array(GRAVITY[activity["posture"]][loc])
    amp = activity[f"a_{loc}"] * subj_amp
    f = activity["f"] * subj_freq

    if f > 0:  # activité périodique : pas de marche/course/pédalage
        phase = RNG.uniform(0, 2 * np.pi, 3)
        motion = np.stack([
            amp * np.sin(2 * np.pi * f * t + phase[0]),
            0.6 * amp * np.sin(2 * np.pi * 2 * f * t + phase[1]),  # harmonique
            0.4 * amp * np.sin(2 * np.pi * f * t + phase[2]),
        ], axis=1)
        gyro = np.stack([
            0.8 * amp * np.cos(2 * np.pi * f * t + phase[0]),
            0.3 * amp * np.sin(2 * np.pi * f * t + phase[1]),
            0.2 * amp * np.cos(2 * np.pi * 2 * f * t + phase[2]),
        ], axis=1)
    else:      # posture statique : micro-mouvements
        motion = RNG.normal(0, amp, (n, 3))
        gyro = RNG.normal(0, amp * 0.5, (n, 3))

    noise = RNG.normal(0, 0.15, (n, 3))
    acc16 = g + motion + noise
    acc6 = np.clip(acc16 + RNG.normal(0, 0.05, (n, 3)), -6 * 9.81, 6 * 9.81)
    mag = RNG.normal([20, -32, 44], 1.5, (n, 3))
    temp = np.full((n, 1), 31.0 + RNG.normal(0, 0.5)) + \
        np.linspace(0, 0.3, n).reshape(-1, 1)
    orient = np.zeros((n, 4))  # invalide dans le vrai jeu -> 0
    return np.hstack([temp, acc16, acc6, gyro, mag, orient])


def _heart_rate(n: int, hr_base: float, subj_hr: float) -> np.ndarray:
    """Cardio ~9 Hz : une mesure tous les ~11 échantillons, NaN sinon."""
    hr = np.full(n, np.nan)
    idx = np.arange(0, n, 11)
    drift = np.cumsum(RNG.normal(0, 0.15, len(idx)))
    hr[idx] = hr_base + subj_hr + drift + RNG.normal(0, 1.0, len(idx))
    return hr


def generate() -> None:
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    activities = [int(a) for a in cfg["activities"].keys()]

    for s in range(9):
        subject_id = 101 + s
        # Variabilité inter-sujets : morphologie, démarche, condition physique
        subj_amp = RNG.uniform(0.75, 1.30)
        subj_freq = RNG.uniform(0.90, 1.12)
        subj_hr = RNG.normal(0, 8)

        blocks, t0 = [], 0.0
        order = RNG.permutation(activities)
        for act_id in order:
            prof = ACTIVITY_PROFILES[act_id]
            # -- transition (activityID = 0) --
            nt = TRANSIENT_SECONDS * FS
            trans = np.hstack([
                (t0 + np.arange(nt) / FS).reshape(-1, 1),
                np.zeros((nt, 1)),
                _heart_rate(nt, prof["hr"] * 0.9, subj_hr).reshape(-1, 1),
                _imu_block(nt, ACTIVITY_PROFILES[3], "hand", subj_amp, subj_freq),
                _imu_block(nt, ACTIVITY_PROFILES[3], "chest", subj_amp, subj_freq),
                _imu_block(nt, ACTIVITY_PROFILES[3], "ankle", subj_amp, subj_freq),
            ])
            blocks.append(trans); t0 += TRANSIENT_SECONDS
            # -- activité --
            n = SECONDS_PER_ACTIVITY * FS
            block = np.hstack([
                (t0 + np.arange(n) / FS).reshape(-1, 1),
                np.full((n, 1), act_id),
                _heart_rate(n, prof["hr"], subj_hr).reshape(-1, 1),
                _imu_block(n, prof, "hand", subj_amp, subj_freq),
                _imu_block(n, prof, "chest", subj_amp, subj_freq),
                _imu_block(n, prof, "ankle", subj_amp, subj_freq),
            ])
            blocks.append(block); t0 += SECONDS_PER_ACTIVITY

        data = np.vstack(blocks)
        # Pertes sans fil : 0,4 % de NaN isolés + quelques rafales de 30-80 éch.
        imu_cols = slice(3, 54)
        mask = RNG.random(data[:, imu_cols].shape) < 0.004
        sub = data[:, imu_cols]; sub[mask] = np.nan; data[:, imu_cols] = sub
        for _ in range(6):
            start = RNG.integers(0, len(data) - 100)
            length = RNG.integers(30, 80)
            col = RNG.integers(3, 54)
            data[start:start + length, col] = np.nan

        path = out_dir / f"subject{subject_id}.dat"
        np.savetxt(path, data, fmt="%.4f")
        log.info("subject%d.dat : %d lignes (%.0f s), %.1f%% NaN IMU",
                 subject_id, len(data), len(data) / FS,
                 100 * np.isnan(data[:, imu_cols]).mean())

    log.info("9 sujets générés dans %s", out_dir)


if __name__ == "__main__":
    generate()
