"""
ÉTAPE T (Transform) — Nettoyage des signaux, spécifique séries temporelles.

Différence majeure avec des données tabulaires classiques : on ne peut pas
« supprimer les lignes NaN » n'importe comment, car cela CASSERAIT la
continuité temporelle du signal (le fenêtrage suppose des échantillons
contigus à 100 Hz). On INTERPOLE donc les petits trous, et on ne supprime
que ce qui est structurellement inutilisable.

Règles (toutes traçables dans les logs) :
  1. activityID = 0 (transitions) -> retiré : étiquette sans signification
  2. colonnes d'orientation      -> retirées : invalides (doc officielle)
  3. cardio ~9 Hz                -> interpolation linéaire par sujet
  4. trous IMU <= 0,2 s          -> interpolation linéaire par sujet
     (les trous plus longs restent NaN ; la segmentation rejettera
      les fenêtres trop incomplètes — voir min_valid_ratio)
"""
from __future__ import annotations

import pandas as pd

from src.config import get_logger, load_config, resolve
from src.ingestion import extract

log = get_logger("cleaning")


def clean(df: pd.DataFrame | None = None, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    rules = cfg["cleaning"]
    if df is None:
        df = extract(cfg)

    n0 = len(df)
    log.info("Nettoyage — %d lignes en entrée", n0)

    # 1. Transitions entre activités
    if rules["drop_transient"]:
        n_trans = int((df["activity_id"] == 0).sum())
        df = df[df["activity_id"] != 0]
        log.info("  - transitions (activity=0)        : %7d lignes retirées", n_trans)

    # Activités hors périmètre (le vrai jeu contient des activités optionnelles)
    keep = [int(a) for a in cfg["activities"].keys()]
    n_other = int((~df["activity_id"].isin(keep)).sum())
    if n_other:
        df = df[df["activity_id"].isin(keep)]
        log.info("  - activités hors périmètre        : %7d lignes retirées", n_other)

    # 2. Colonnes d'orientation (invalides selon la documentation officielle)
    if rules["drop_orientation"]:
        orient = [c for c in df.columns if "orient" in c]
        df = df.drop(columns=orient)
        log.info("  - colonnes orientation            : %7d colonnes retirées", len(orient))

    # 3 & 4. Interpolation PAR SUJET (jamais à travers deux sujets !)
    df = df.copy()
    sensor_cols = [c for c in df.columns if c not in
                   ("subject_id", "timestamp", "activity_id")]
    max_gap = rules["max_gap_interpolate"]

    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["heart_rate"] = g["heart_rate"].interpolate(
            method=rules["hr_interpolation"], limit_direction="both")
        imu = [c for c in sensor_cols if c != "heart_rate"]
        g[imu] = g[imu].interpolate(method="linear", limit=max_gap,
                                    limit_direction="both")
        return g

    nan_before = df[sensor_cols].isna().to_numpy().mean()
    # include_groups=False retire subject_id du résultat ; on le réinjecte via
    # l'INDEX (et non par position) pour rester correct même si l'ordre des
    # lignes changeait après le groupby.
    out = df.groupby("subject_id", group_keys=False).apply(
        _interp, include_groups=False)
    out["subject_id"] = df["subject_id"].reindex(out.index)
    df = out
    nan_after = df[sensor_cols].isna().to_numpy().mean()
    log.info("  - interpolation par sujet         : NaN %.2f%% -> %.2f%%",
             100 * nan_before, 100 * nan_after)

    df["activity_id"] = df["activity_id"].astype(int)
    log.info("Nettoyage terminé : %d -> %d lignes (%.1f%% conservées)",
             n0, len(df), 100 * len(df) / n0)

    out = resolve(cfg["paths"]["interim"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Signaux propres sauvegardés : %s", out)
    return df


if __name__ == "__main__":
    clean()
