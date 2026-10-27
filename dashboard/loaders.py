"""Accès aux données et artefacts du pipeline, mis en cache pour toutes les pages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.ingestion import COLUMNS

cfg = load_config()
ACTIVITIES = {int(k): v for k, v in cfg["activities"].items()}
ACTIVITY_LABELS = {0: "transition", **ACTIVITIES}

# Couches du pipeline : clé de config -> (libellé, rôle)
LAYERS = {
    "raw": ("Brute", "Un fichier .dat par sujet, sans en-tête : 54 colonnes séparées par des espaces, "
                     "100 Hz, transitions incluses (activité 0)."),
    "interim": ("Nettoyée", "Sujets consolidés : transitions retirées, colonnes d'orientation invalides "
                            "supprimées, valeurs manquantes interpolées par sujet. Format Parquet."),
    "features": ("Features", "Une ligne par fenêtre de 2,56 s (recouvrement 50 %) : statistiques "
                             "temporelles et fréquentielles par canal. Jeu d'entrée du modèle."),
}


def raw_files() -> list[str]:
    return sorted(p.name for p in resolve(cfg["paths"]["raw_dir"]).glob("subject*.dat"))


@st.cache_data
def load_layer(name: str, subject_file: str | None = None) -> pd.DataFrame:
    if name == "raw":
        df = pd.read_csv(resolve(cfg["paths"]["raw_dir"]) / subject_file, sep=" ", header=None, names=COLUMNS)
        return df.assign(subject_id=int(subject_file[7:10]))
    return pd.read_parquet(resolve(cfg["paths"][name]))


@st.cache_data
def layer_stats(name: str) -> dict:
    """Chiffres clés d'une couche ; la couche brute cumule tous les fichiers .dat."""
    if name == "raw":
        raw_dir = resolve(cfg["paths"]["raw_dir"])
        frames = [load_layer("raw", f) for f in raw_files()]
        rows = sum(len(f) for f in frames)
        return {"rows": rows, "cols": len(COLUMNS), "format": "DAT", "files": len(frames),
                "missing": sum(f[COLUMNS].isna().sum().sum() for f in frames) / max(1, rows * len(COLUMNS)),
                "size_mb": sum((raw_dir / f).stat().st_size for f in raw_files()) / 1e6}
    df = load_layer(name)
    path = resolve(cfg["paths"][name])
    return {"rows": len(df), "cols": df.shape[1], "size_mb": path.stat().st_size / 1e6, "files": 1,
            "missing": df.isna().mean().mean(), "format": path.suffix[1:].upper()}


@st.cache_data
def column_profile(name: str, subject_file: str | None = None) -> pd.DataFrame:
    df = load_layer(name, subject_file)
    return pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Manquants (%)": (df.isna().mean() * 100).round(1).values,
        "Valeurs distinctes": df.nunique().values,
        "Exemple": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
    })


@st.cache_data
def load_metrics() -> dict:
    path = resolve(cfg["paths"]["metrics"])
    return json.loads(path.read_text()) if path.exists() else {}


def figure_path(name: str) -> Path:
    return resolve(cfg["paths"]["figures"]) / name


MODEL_NAMES = {"xgboost": "XGBoost", "logistic_regression": "Régression logistique",
               "random_forest": "Random Forest"}


def champion_label(metrics: dict) -> str:
    name = metrics.get("champion_deployed", "—")
    return MODEL_NAMES.get(name, name.replace("_", " ").title())
