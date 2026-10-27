import pandas as pd
import streamlit as st

import ui
from loaders import champion_label, figure_path, load_metrics

metrics = load_metrics()

ui.page_header("Modèle", "Performance du modèle",
               f"Évaluation de {champion_label(metrics)} sur {ui.fmt_int(metrics.get('n_test_windows', 0))} "
               f"fenêtres issues de sujets jamais vus à l'entraînement.")

if not metrics:
    st.warning("Aucune métrique trouvée : lancez `python run_pipeline.py`.")
    st.stop()

ui.kpi_row([
    ("F1-macro test", f"{metrics['f1_macro_test_subject_wise']:.3f}",
     f"CV groupée : {metrics['f1_macro_cv_grouped']:.3f}", True),
    ("Exactitude test", f"{metrics['accuracy_test']:.1%}"),
    ("Split naïf (biaisé)", f"{metrics['naive_split_f1_LEAKY']:.3f}", f"écart {metrics['leakage_gap']:+.3f}"),
    ("Sujets test", ", ".join(map(str, metrics.get("test_subjects", []))),
     f"entraînement : {len(metrics.get('train_subjects', []))} sujets"),
])

ui.section("Sélection du modèle", "Le champion est choisi sur la validation croisée groupée par sujet.")
baseline = metrics.get("baseline_logistic_regression", {})
baseline_wins = metrics["champion_deployed"] == "logistic_regression"
st.dataframe(pd.DataFrame({
    "Candidat": ["Régression logistique (baseline)", "Challenger (Random Forest / XGBoost)"],
    "F1-macro CV": [baseline.get("f1_cv"), metrics.get("challenger_rf_or_xgb_cv", metrics.get("challenger_cv"))],
    "Retenu": [baseline_wins, not baseline_wins],
}), hide_index=True, width="stretch",
    column_config={"F1-macro CV": st.column_config.NumberColumn(format="%.4f")})
if baseline_wins:
    ui.note("Le modèle linéaire généralise mieux aux nouveaux sujets que le challenger plus complexe : "
            "la complexité supplémentaire n'est pas justifiée ici.")

ui.section("Diagnostics")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("confusion_matrix.png"), "Matrice de confusion (sujets test)")
with c2:
    ui.figure(figure_path("leakage_comparison.png"), "Split par sujet vs split aléatoire")
ui.figure(figure_path("feature_importance.png"), "Importance des variables")

with st.expander("Métriques brutes (metrics.json)"):
    st.json(metrics)
