"""
ÉTAPE E (Extract) — Acquisition + ingestion des fichiers .dat PAMAP2.

Sources (config data_source.mode) :
  - "uci"   : télécharge le zip officiel depuis l'UCI ML Repository,
              extrait les fichiers Protocol/*.dat. Repli local si échec.
  - "local" : lit directement data/raw/Protocol/*.dat.

Les 54 colonnes suivent la spécification officielle du dataset.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve

log = get_logger("ingestion")

# --- Noms des 54 colonnes (spécification PAMAP2) -----------------------------
IMU_FIELDS = (["temp"]
              + [f"acc16_{a}" for a in "xyz"] + [f"acc6_{a}" for a in "xyz"]
              + [f"gyro_{a}" for a in "xyz"] + [f"mag_{a}" for a in "xyz"]
              + [f"orient_{i}" for i in range(1, 5)])
COLUMNS = (["timestamp", "activity_id", "heart_rate"]
           + [f"hand_{f}" for f in IMU_FIELDS]
           + [f"chest_{f}" for f in IMU_FIELDS]
           + [f"ankle_{f}" for f in IMU_FIELDS])
assert len(COLUMNS) == 54


def acquire(cfg: dict) -> Path:
    """S'assure que les .dat sont présents ; les télécharge si demandé."""
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    have_files = raw_dir.exists() and any(raw_dir.glob("subject*.dat"))
    mode = cfg.get("data_source", {}).get("mode", "local")

    if mode == "uci" and not have_files:
        url = cfg["data_source"]["uci_url"]
        try:
            log.info("Téléchargement PAMAP2 depuis l'UCI (~600 Mo)...")
            with urllib.request.urlopen(url, timeout=120) as r:
                outer = zipfile.ZipFile(io.BytesIO(r.read()))
            # Le zip UCI contient PAMAP2_Dataset.zip qui contient Protocol/
            inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
            inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
            raw_dir.mkdir(parents=True, exist_ok=True)
            for name in inner.namelist():
                if "/Protocol/subject" in name and name.endswith(".dat"):
                    (raw_dir / Path(name).name).write_bytes(inner.read(name))
            log.info("Fichiers Protocol extraits dans %s", raw_dir)
        except Exception as e:
            log.warning("Téléchargement UCI impossible (%s).", e)
            if not have_files:
                raise RuntimeError(
                    "Pas de données. Solutions : (1) vérifier la connexion, "
                    "(2) générer un jeu de démo : python -m src.generate_sample_data, "
                    "(3) déposer les subject*.dat dans data/raw/Protocol/."
                ) from e
            log.warning("Repli sur les fichiers locaux existants.")
    return raw_dir


def extract(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_dir = acquire(cfg)
    files = sorted(raw_dir.glob("subject*.dat"))
    if not files:
        raise FileNotFoundError(f"Aucun subject*.dat dans {raw_dir}")

    frames = []
    for f in files:
        subject_id = int(f.stem.replace("subject", ""))
        arr = np.loadtxt(f)                       # NaN gérés nativement
        df = pd.DataFrame(arr, columns=COLUMNS)
        df.insert(0, "subject_id", subject_id)
        frames.append(df)
        log.info("%s : %d lignes", f.name, len(df))
    df = pd.concat(frames, ignore_index=True)

    # --- Rapport qualité ---
    imu_cols = [c for c in COLUMNS if c not in
                ("timestamp", "activity_id", "heart_rate")]
    log.info("Total : %d lignes | %d sujets | %d activités distinctes",
             len(df), df["subject_id"].nunique(),
             df.loc[df["activity_id"] > 0, "activity_id"].nunique())
    log.info("NaN cardio : %.1f%% (normal : ~9 Hz vs 100 Hz IMU)",
             100 * df["heart_rate"].isna().mean())
    log.info("NaN IMU    : %.2f%% (pertes sans fil)",
             100 * df[imu_cols].isna().to_numpy().mean())
    log.info("Transitoire (activity=0) : %.1f%% des lignes",
             100 * (df["activity_id"] == 0).mean())
    return df


if __name__ == "__main__":
    extract()
