"""
ÉTAPE Modélisation — Classification multi-classes des activités.

LE point méthodologique de ce projet : la SÉPARATION PAR SUJET.

Deux fenêtres consécutives du même sujet se ressemblent énormément (même
démarche, même morphologie, 50 % de chevauchement). Si on coupe train/test
au hasard sur les fenêtres, le modèle « reconnaît la personne » au lieu de
« reconnaître l'activité » -> score gonflé, illusoire (fuite de données).

La bonne question est : « le modèle marche-t-il sur une personne JAMAIS
VUE ? ». On sépare donc par SUJET : ~6 sujets pour apprendre, ~3 pour
tester. C'est le protocole de la littérature HAR (leave-subjects-out).

Pour le PROUVER, on calcule les deux scores et on log l'écart : la
démonstration chiffrée de la fuite vaut tous les discours en soutenance.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_logger, load_config, resolve

log = get_logger("train")


def _build_estimator(cfg: dict):
    algo = cfg["model"]["algorithm"]
    rs = cfg["project"]["random_state"]
    if algo == "random_forest":
        base = RandomForestClassifier(random_state=rs, n_jobs=-1,
                                      class_weight="balanced")
        grid = {f"model__{k}": v for k, v in cfg["model"]["random_forest"].items()}
    elif algo == "xgboost":
        from xgboost import XGBClassifier
        base = XGBClassifier(random_state=rs, n_jobs=-1, tree_method="hist",
                             eval_metric="mlogloss")
        grid = {f"model__{k}": v for k, v in cfg["model"]["xgboost"].items()}
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")
    return base, grid


def train(cfg: dict | None = None):
    cfg = cfg or load_config()
    rs = cfg["project"]["random_state"]
    df = pd.read_parquet(resolve(cfg["paths"]["features"]))

    feat_cols = [c for c in df.columns if c not in ("subject_id", "activity_id")]
    X, y, groups = df[feat_cols], df["activity_id"], df["subject_id"]

    # ------------------------------------------------------------------
    # DÉMONSTRATION DE LA FUITE : split naïf (fenêtres au hasard)
    # IMPORTANT : on tient l'ALGORITHME CONSTANT (LogisticRegression, le même
    # que la baseline honnête ci-dessous). L'écart naïf vs honnête mesure alors
    # la SEULE fuite de données, pas un changement de modèle.
    # ------------------------------------------------------------------
    def _logreg():
        return Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=2000,
                                                      random_state=rs))])

    naive_model = _logreg()
    naive_test = cfg["model"].get("naive_test_size", 0.3)
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        X, y, test_size=naive_test, stratify=y, random_state=rs)
    naive_model.fit(Xn_tr, yn_tr)
    naive_f1 = f1_score(yn_te, naive_model.predict(Xn_te), average="macro")
    log.info("SPLIT NAÏF  (fenêtres au hasard, MÊME algo)  : F1-macro = %.3f"
             "  <- GONFLÉ (fuite)", naive_f1)

    # ------------------------------------------------------------------
    # SPLIT HONNÊTE : par sujet (sujets de test jamais vus)
    # ------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=cfg["model"]["test_subjects_ratio"],
                            random_state=rs)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    g_tr = groups.iloc[tr_idx]
    subj_tr = sorted(groups.iloc[tr_idx].unique())
    subj_te = sorted(groups.iloc[te_idx].unique())
    log.info("SPLIT PAR SUJET — train : %s | test (jamais vus) : %s",
             subj_tr, subj_te)

    # Baseline : régression logistique multinomiale (même split honnête)
    cv_test = cfg["model"].get("cv_test_size", 0.25)
    baseline = _logreg()
    bl_cv = cross_val_score(baseline, X_tr, y_tr, scoring="f1_macro",
                            cv=GroupShuffleSplit(n_splits=cfg["model"]["cv_folds"],
                                                 test_size=cv_test, random_state=rs),
                            groups=g_tr, n_jobs=-1).mean()
    baseline.fit(X_tr, y_tr)
    bl_test = f1_score(y_te, baseline.predict(X_te), average="macro")
    log.info("BASELINE LogisticRegression — F1-macro CV : %.3f | test : %.3f",
             bl_cv, bl_test)

    # Modèle principal : validation croisée GROUPÉE par sujet aussi
    base, grid = _build_estimator(cfg)
    pipe = Pipeline([("scale", StandardScaler()), ("model", base)])
    group_cv = GroupShuffleSplit(n_splits=cfg["model"]["cv_folds"],
                                 test_size=cv_test, random_state=rs)
    search = GridSearchCV(pipe, grid, scoring="f1_macro",
                          cv=group_cv.split(X_tr, y_tr, g_tr), n_jobs=-1)
    log.info("Recherche d'hyperparamètres (CV groupée par sujet)...")
    search.fit(X_tr, y_tr)
    log.info("Meilleurs paramètres : %s", search.best_params_)
    log.info("F1-macro (CV groupée) : %.3f", search.best_score_)

    best = search.best_estimator_

    # ------------------------------------------------------------------
    # SÉLECTION DU CHAMPION : on déploie le meilleur en CV GROUPÉE.
    # Si le modèle simple généralise mieux inter-sujets que le modèle
    # complexe, c'est LUI qu'on met en production (plus léger, plus
    # interprétable). La complexité doit se mériter.
    # ------------------------------------------------------------------
    if bl_cv >= search.best_score_:
        champion, champ_name, champ_cv = baseline, "logistic_regression", bl_cv
    else:
        champion, champ_name, champ_cv = best, cfg["model"]["algorithm"], search.best_score_
    log.info("CHAMPION déployé : %s (F1-macro CV groupée = %.3f)",
             champ_name, champ_cv)

    joblib.dump({"model": champion, "feat_cols": feat_cols,
                 "activities": cfg["activities"], "champion": champ_name},
                resolve(cfg["paths"]["model"]))
    log.info("Modèle sauvegardé : %s", resolve(cfg["paths"]["model"]))

    # best_params : uniquement si le challenger (RF/XGB) est réellement déployé.
    # Sinon {} pour ne pas afficher des hyperparamètres qui ne sont PAS ceux
    # du modèle en production (la baseline LogReg n'a pas de grille).
    champ_params = search.best_params_ if champ_name != "logistic_regression" else {}
    extras = {
        "naive_split_f1": round(float(naive_f1), 4),
        "baseline": {"f1_cv": round(float(bl_cv), 4),
                     "f1_test": round(float(bl_test), 4)},
        "champion": champ_name,
        "challenger_rf_or_xgb_cv": round(float(search.best_score_), 4),
        "challenger_best_params": search.best_params_,
        "train_subjects": [int(s) for s in subj_tr],
        "test_subjects": [int(s) for s in subj_te],
    }
    return champion, X_te, y_te, champ_params, champ_cv, extras


if __name__ == "__main__":
    train()
