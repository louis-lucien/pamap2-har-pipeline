"""
Point d'entrée principal — tout le pipeline en une commande.

    python run_pipeline.py

Enchaîne : acquisition -> nettoyage -> segmentation/features -> EDA
           -> baseline + champion -> évaluation sur sujets jamais vus.
"""
from __future__ import annotations

import time

from src.cleaning import clean
from src.config import get_logger, load_config
from src.eda import run_eda
from src.evaluate import evaluate
from src.features import build_features
from src.ingestion import extract
from src.train import train

log = get_logger("pipeline")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("=== DÉBUT DU PIPELINE : %s v%s ===",
             cfg["project"]["name"], cfg["project"]["version"])

    raw = extract(cfg)                                   # 1. Extract (.dat)
    clean_df = clean(raw, cfg)                           # 2. Nettoyage signal
    build_features(clean_df, cfg)                        # 3. Fenêtres + features
    run_eda(cfg)                                         # 4. EDA signaux
    model, X_te, y_te, bp, cv_f1, extras = train(cfg)    # 5. Baseline + champion
    metrics = evaluate(model, X_te, y_te, bp, cv_f1, extras, cfg)

    log.info("=== FIN en %.1fs — F1-macro (sujets jamais vus) = %.3f | "
             "champion : %s ===", time.time() - t0,
             metrics["f1_macro_test_subject_wise"], metrics["champion_deployed"])


if __name__ == "__main__":
    main()
