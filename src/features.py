"""
ÉTAPE Segmentation + Extraction de caractéristiques.

Principe (analogie Shazam) : on ne classe pas un échantillon isolé — un seul
point d'accéléromètre ne dit rien — mais une FENÊTRE de 2,56 s (256 éch. à
100 Hz), comme Shazam reconnaît une chanson sur un extrait de 3 s. Chaque
fenêtre est résumée en un vecteur de caractéristiques :

  Domaine temporel  : mean, std, min, max, médiane, énergie   (par canal)
  Inter-axes        : magnitude ||a|| par IMU (invariante à l'orientation)
  Domaine fréquentiel : fréquence dominante par FFT (la cadence des pas !)
  Physiologique     : cardio moyen et écart-type

La fonction extract_features_from_window() est LA MÊME en entraînement et
dans l'API : aucun écart possible entre le labo et la production.

Fenêtres à 50 % de recouvrement, mais JAMAIS à cheval sur deux activités ou
deux sujets (on segmente à l'intérieur des blocs contigus).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("features")

# Canaux utilisés pour les features. acc16 plutôt qu'acc6 : la doc officielle
# recommande ±16g car ±6g sature pendant la course.
ACC_CHANNELS = [f"{loc}_acc16_{a}" for loc in ("hand", "chest", "ankle")
                for a in "xyz"]
GYRO_CHANNELS = [f"{loc}_gyro_{a}" for loc in ("hand", "chest", "ankle")
                 for a in "xyz"]


def extract_features_from_window(w: pd.DataFrame, fs: int = 100) -> dict:
    """Résume une fenêtre (DataFrame de ~256 lignes) en un vecteur de features."""
    feats: dict[str, float] = {}

    for c in ACC_CHANNELS + GYRO_CHANNELS:
        x = w[c].to_numpy()
        feats[f"{c}_mean"] = float(np.nanmean(x))
        feats[f"{c}_std"] = float(np.nanstd(x))
        feats[f"{c}_min"] = float(np.nanmin(x))
        feats[f"{c}_max"] = float(np.nanmax(x))
        feats[f"{c}_median"] = float(np.nanmedian(x))
        feats[f"{c}_energy"] = float(np.nanmean(x ** 2))

    # Magnitude d'accélération par IMU : ||a|| = sqrt(x²+y²+z²).
    # Invariante à l'orientation du capteur -> robuste au placement.
    for loc in ("hand", "chest", "ankle"):
        axes = w[[f"{loc}_acc16_{a}" for a in "xyz"]].to_numpy()
        mag = np.sqrt(np.nansum(axes ** 2, axis=1))
        feats[f"{loc}_accmag_mean"] = float(np.nanmean(mag))
        feats[f"{loc}_accmag_std"] = float(np.nanstd(mag))
        # Fréquence dominante (FFT) : capture la CADENCE (pas, pédalage).
        # C'est la feature qui distingue marche (~1,9 Hz) et course (~2,9 Hz).
        sig = mag - np.nanmean(mag)
        sig = np.nan_to_num(sig)
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
        band = (freqs >= 0.3) & (freqs <= 10)   # bande utile mouvement humain
        feats[f"{loc}_dom_freq"] = (float(freqs[band][np.argmax(spectrum[band])])
                                    if band.any() else 0.0)

    hr = w["heart_rate"].to_numpy()
    feats["hr_mean"] = float(np.nanmean(hr))
    feats["hr_std"] = float(np.nanstd(hr))
    return feats


def build_features(df: pd.DataFrame | None = None,
                   cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_parquet(resolve(cfg["paths"]["interim"]))

    seg = cfg["segmentation"]
    win, fs = seg["window_size"], cfg["sensors"]["sampling_rate_hz"]
    step = int(win * (1 - seg["overlap"]))
    min_valid = seg["min_valid_ratio"]

    rows, rejected = [], 0
    # Blocs contigus : même sujet ET même activité (jamais de fenêtre à cheval)
    block_id = ((df["subject_id"].diff() != 0) |
                (df["activity_id"].diff() != 0)).cumsum()
    for _, block in df.groupby(block_id):
        if len(block) < win:
            continue
        sensor_cols = ACC_CHANNELS + GYRO_CHANNELS
        for start in range(0, len(block) - win + 1, step):
            w = block.iloc[start:start + win]
            valid_ratio = 1 - w[sensor_cols].isna().to_numpy().mean()
            if valid_ratio < min_valid:
                rejected += 1
                continue
            feats = extract_features_from_window(w, fs)
            feats["subject_id"] = int(w["subject_id"].iloc[0])
            feats["activity_id"] = int(w["activity_id"].iloc[0])
            rows.append(feats)

    feats_df = pd.DataFrame(rows)
    n_feat = len([c for c in feats_df.columns
                  if c not in ("subject_id", "activity_id")])
    log.info("Fenêtres : %d retenues, %d rejetées (NaN > %.0f%%)",
             len(feats_df), rejected, 100 * (1 - min_valid))
    log.info("Features par fenêtre : %d | Répartition activités :\n%s",
             n_feat, feats_df["activity_id"].value_counts().sort_index().to_string())

    out = resolve(cfg["paths"]["features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    feats_df.to_parquet(out, index=False)
    log.info("Table de features sauvegardée : %s", out)
    return feats_df


if __name__ == "__main__":
    build_features()
