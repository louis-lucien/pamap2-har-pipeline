"""
ÉTAPE Inférence — Prédire l'activité depuis une FENÊTRE BRUTE de capteurs.

Point d'architecture important : l'API reçoit du signal brut (listes
d'échantillons) et calcule les features avec EXACTEMENT la même fonction
que l'entraînement (src.features.extract_features_from_window).
=> Aucun écart labo/production possible (pas de training-serving skew).
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve
from src.features import ACC_CHANNELS, GYRO_CHANNELS, extract_features_from_window

log = get_logger("predict")


@lru_cache(maxsize=1)
def _load():
    cfg = load_config()
    bundle = joblib.load(resolve(cfg["paths"]["model"]))
    log.info("Modèle chargé (champion : %s)", bundle.get("champion", "?"))
    return bundle


def predict_window(channels: dict, heart_rate: list | None = None) -> dict:
    """channels : dict {nom_canal: liste d'échantillons} pour les 18 canaux
    acc16/gyro des 3 IMU. heart_rate : liste optionnelle (peut contenir None)."""
    bundle = _load()
    cfg = load_config()
    fs = cfg["sensors"]["sampling_rate_hz"]

    n = len(next(iter(channels.values())))
    data = {}
    for c in ACC_CHANNELS + GYRO_CHANNELS:
        if c not in channels:
            raise ValueError(f"Canal manquant : {c}")
        if len(channels[c]) != n:
            raise ValueError(f"Longueur incohérente pour {c}")
        data[c] = np.asarray(channels[c], dtype=float)
    hr = ([np.nan if v is None else float(v) for v in heart_rate]
          if heart_rate else [np.nan] * n)
    data["heart_rate"] = np.asarray(hr, dtype=float)

    w = pd.DataFrame(data)
    feats = extract_features_from_window(w, fs)
    X = pd.DataFrame([feats])[bundle["feat_cols"]].fillna(0.0)

    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    pred = int(classes[int(np.argmax(proba))])
    names = {int(k): v for k, v in bundle["activities"].items()}
    top3 = sorted(zip(classes, proba), key=lambda t: -t[1])[:3]
    return {
        "activity_id": pred,
        "activity_name": names.get(pred, str(pred)),
        "confidence": round(float(proba.max()), 4),
        "top3": [{"activity": names.get(int(c), str(c)),
                  "probability": round(float(p), 4)} for c, p in top3],
        "champion_model": bundle.get("champion", "?"),
    }


if __name__ == "__main__":
    # Démo : fenêtre extraite du jeu de test
    cfg = load_config()
    sig = pd.read_parquet(resolve(cfg["paths"]["interim"]))
    w = sig[(sig["subject_id"] == 102) & (sig["activity_id"] == 5)].iloc[:256]
    channels = {c: w[c].tolist() for c in ACC_CHANNELS + GYRO_CHANNELS}
    res = predict_window(channels, w["heart_rate"].tolist())
    print("Vérité : course |", res)
