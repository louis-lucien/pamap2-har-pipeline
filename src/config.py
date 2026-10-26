"""
Chargement de la configuration et du logging.

Un seul point d'entrée pour lire config/config.yaml. Toutes les autres
étapes du pipeline importent `load_config()` — aucune valeur n'est
codée en dur ailleurs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

# Racine du projet = dossier parent de src/
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Lit le YAML de configuration et renvoie un dictionnaire."""
    cfg_path = ROOT / path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(path_str: str) -> Path:
    """Transforme un chemin relatif de la config en chemin absolu."""
    return ROOT / path_str


def get_logger(name: str) -> logging.Logger:
    """Logger uniforme pour toutes les étapes (horodaté, lisible)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
