"""
Orchestration Prefect — 6 tâches, retries sur l'acquisition.

Lancer :  python -m pipelines.flow
"""
from __future__ import annotations

from prefect import flow, task

from src.cleaning import clean
from src.config import load_config
from src.eda import run_eda
from src.evaluate import evaluate
from src.features import build_features
from src.ingestion import extract
from src.train import train


@task(name="extract", retries=2, retry_delay_seconds=15)
def t_extract(cfg):
    return extract(cfg)


@task(name="clean")
def t_clean(df, cfg):
    return clean(df, cfg)


@task(name="features")
def t_features(df, cfg):
    return build_features(df, cfg)


@task(name="eda")
def t_eda(cfg):
    return run_eda(cfg)


@task(name="train")
def t_train(cfg):
    return train(cfg)


@task(name="evaluate")
def t_evaluate(model, X_te, y_te, bp, cv_f1, extras, cfg):
    return evaluate(model, X_te, y_te, bp, cv_f1, extras, cfg)


@flow(name="pamap2-har-pipeline", log_prints=True)
def run_pipeline():
    cfg = load_config()
    raw = t_extract(cfg)
    clean_df = t_clean(raw, cfg)
    t_features(clean_df, cfg)
    t_eda(cfg)
    model, X_te, y_te, bp, cv_f1, extras = t_train(cfg)
    metrics = t_evaluate(model, X_te, y_te, bp, cv_f1, extras, cfg)
    print("F1-macro (sujets jamais vus) =", metrics["f1_macro_test_subject_wise"])
    return metrics


if __name__ == "__main__":
    run_pipeline()
