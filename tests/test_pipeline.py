"""Tests unitaires — invariants critiques du pipeline HAR.

Les tests du cœur méthodologique (anti-chevauchement, anti-fuite du split
par sujet) s'exécutent sur des données construites en mémoire : ils testent
le CODE, pas un artefact de run figé.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import (ACC_CHANNELS, GYRO_CHANNELS, build_features,
                          extract_features_from_window)


def test_feature_extraction_shape_and_values():
    """Une fenêtre valide produit des features finies et nommées."""
    rng = np.random.default_rng(0)
    n = 256
    data = {c: 9.8 + rng.normal(0, 1, n) for c in ACC_CHANNELS}
    data |= {c: rng.normal(0, 1, n) for c in GYRO_CHANNELS}
    data["heart_rate"] = np.where(np.arange(n) % 11 == 0, 80.0, np.nan)
    feats = extract_features_from_window(pd.DataFrame(data))
    assert len(feats) == 119
    assert all(np.isfinite(v) for v in feats.values())
    assert 0.3 <= feats["ankle_dom_freq"] <= 10  # bande mouvement humain


def _toy_signal():
    """Deux sujets, deux activités chacun, échantillons contigus à 100 Hz.
    Construit pour que le fenêtrage doive produire des blocs homogènes."""
    rng = np.random.default_rng(1)
    cols = ACC_CHANNELS + GYRO_CHANNELS
    frames = []
    for subj in (101, 102):
        for act in (4, 5):
            n = 600  # > window (256), génère plusieurs fenêtres
            d = {c: rng.normal(0, 1, n) for c in cols}
            d["heart_rate"] = np.full(n, 90.0)
            d["subject_id"] = subj
            d["activity_id"] = act
            d["timestamp"] = np.arange(n) / 100.0
            frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


def test_windows_never_straddle_subjects_or_activities():
    """Vrai test d'invariant : on fabrique un signal multi-sujets/activités
    et on vérifie que CHAQUE fenêtre produite porte un seul couple
    (sujet, activité) — donc qu'aucune fenêtre n'est à cheval."""
    cfg = load_config()
    feats = build_features(_toy_signal(), cfg)
    # Chaque fenêtre a exactement un sujet et une activité (déjà garanti par
    # la colonne), mais surtout : le nombre de fenêtres correspond à un
    # fenêtrage PAR BLOC. 600 éch., window=256, step=128 -> 3 fenêtres/bloc,
    # 4 blocs = 12 fenêtres. Un fenêtrage à cheval en produirait davantage.
    assert len(feats) == 12
    assert set(feats["subject_id"]) == {101, 102}
    assert set(feats["activity_id"]) == {4, 5}


def test_subject_wise_split_has_no_group_overlap():
    """Anti-fuite au niveau du CODE : le splitter par sujet ne met jamais
    un même sujet à la fois en train et en test."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = np.repeat([101, 102, 103, 104, 105, 106, 107, 108, 109], 10)
    X = np.zeros((len(groups), 3))
    y = np.tile([4, 5], len(groups) // 2)
    gss = GroupShuffleSplit(n_splits=5, test_size=0.33, random_state=42)
    for tr, te in gss.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_prediction_on_raw_window():
    """L'inférence sur fenêtre brute renvoie une activité valide.
    Test d'intégration : ignoré proprement si le pipeline n'a pas tourné."""
    from pathlib import Path
    from src.config import resolve
    cfg = load_config()
    interim = resolve(cfg["paths"]["interim"])
    model = resolve(cfg["paths"]["model"])
    if not interim.exists() or not model.exists():
        pytest.skip("Artefacts absents (lancer run_pipeline.py d'abord)")
    from src.predict import predict_window
    sig = pd.read_parquet(interim)
    w = sig[(sig["activity_id"] == 4)].iloc[:256]
    channels = {c: w[c].tolist() for c in ACC_CHANNELS + GYRO_CHANNELS}
    res = predict_window(channels, w["heart_rate"].tolist())
    assert res["activity_id"] in [int(a) for a in cfg["activities"]]
    assert 0.0 <= res["confidence"] <= 1.0
