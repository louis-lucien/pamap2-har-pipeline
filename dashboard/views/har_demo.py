import streamlit as st

import ui
from loaders import ACTIVITIES, cfg, load_layer, load_metrics
from src.features import ACC_CHANNELS, GYRO_CHANNELS
from src.predict import predict_window

metrics = load_metrics()
signals = load_layer("interim")
window = cfg["segmentation"]["window_size"]

ui.page_header("Modèle", "Prédiction sur une fenêtre",
               "Sélectionnez un sujet de test et une activité : une fenêtre de 2,56 s de signal brut "
               "est extraite et soumise au modèle déployé.")

test_subjects = metrics.get("test_subjects", sorted(signals["subject_id"].unique())[:3])
with st.form("prediction"):
    c1, c2, c3 = st.columns(3)
    subject = c1.selectbox("Sujet (hors entraînement)", test_subjects)
    activity = c2.selectbox("Activité réelle", list(ACTIVITIES), format_func=ACTIVITIES.get)
    offset = c3.number_input("Index de la fenêtre", 0, 12, 3)
    submitted = st.form_submit_button("Lancer la prédiction", type="primary", width="stretch")

if submitted:
    block = signals[(signals["subject_id"] == subject) & (signals["activity_id"] == activity)]
    w = block.iloc[offset * window:(offset + 1) * window]
    if len(w) < window:
        st.warning(f"Fenêtre incomplète ({len(w)} échantillons sur {window}) : choisissez un index plus petit.")
        st.stop()
    result = predict_window({c: w[c].tolist() for c in ACC_CHANNELS + GYRO_CHANNELS}, w["heart_rate"].tolist())
    correct = result["activity_id"] == activity

    ui.section("Résultat")
    ui.kpi_row([
        ("Activité prédite", result["activity_name"], "", True),
        ("Confiance", f"{result['confidence']:.1%}"),
        ("Vérité terrain", ACTIVITIES[activity]),
        ("Verdict", "Correct" if correct else "Erreur"),
    ])
    (st.success if correct else st.error)(
        "Prédiction conforme à la vérité terrain." if correct else "Prédiction différente de la vérité terrain.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Accélération de la cheville (fenêtre analysée)**")
        st.line_chart(w[["ankle_acc16_x", "ankle_acc16_y", "ankle_acc16_z"]].reset_index(drop=True))
    with c2:
        st.markdown("**Top 3 des classes**")
        st.json(result["top3"])
