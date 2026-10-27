import streamlit as st

import ui
from loaders import ACTIVITIES, cfg, champion_label, layer_stats, load_metrics

metrics = load_metrics()
seg = cfg["segmentation"]

ui.page_header("Vue d'ensemble", "Reconnaissance d'activité physique",
               "Classer l'activité d'une personne (marche, course, vélo, escaliers…) à partir de trois "
               "centrales inertielles portées au poignet, à la poitrine et à la cheville, et du rythme cardiaque.")

ui.kpi_row([
    ("Activités", len(ACTIVITIES), "classes cibles", True),
    ("Capteurs", "3 IMU + FC", f"{cfg['sensors']['sampling_rate_hz']} Hz"),
    ("F1-macro test", f"{metrics.get('f1_macro_test_subject_wise', 0):.3f}", "sujets jamais vus", True),
    ("Modèle déployé", champion_label(metrics), "sélection sur CV groupée"),
])

ui.section("Chaîne de traitement", "Du signal brut des capteurs à la prédiction en temps réel.")
raw, interim, feats = (layer_stats(k) for k in ("raw", "interim", "features"))
ui.flow([
    ("Ingestion", ui.fmt_int(raw["rows"]), f"mesures · {raw['files']} fichiers .dat"),
    ("Nettoyage", ui.fmt_int(interim["rows"]), "mesures valides · Parquet"),
    ("Fenêtrage", f"{seg['window_size'] / cfg['sensors']['sampling_rate_hz']:.2f} s",
     f"recouvrement {seg['overlap']:.0%}"),
    ("Features", ui.fmt_int(feats["rows"]), f"fenêtres · {feats['cols'] - 2} variables"),
    ("Modèle", champion_label(metrics), "split par sujet"),
    ("Service", "API + démo", "prédiction par fenêtre"),
])

ui.section("Points clés")
c1, c2 = st.columns(2)
with c1:
    ui.note("<b>Évaluation par sujet.</b> Les sujets de test ne sont jamais vus à l'entraînement : "
            "le score reflète la performance sur un nouvel utilisateur.")
with c2:
    if "naive_split_f1_LEAKY" in metrics:
        ui.note(f"<b>Fuite mesurée.</b> Un split aléatoire donnerait {metrics['naive_split_f1_LEAKY']:.3f} "
                f"(écart {metrics['leakage_gap']:+.3f}) : score surestimé car le modèle reconnaît la personne.")
