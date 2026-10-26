"""
Dashboard Streamlit — signaux, performance, et démo de reconnaissance live.

Lancer :  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import load_config, resolve
from src.features import ACC_CHANNELS, GYRO_CHANNELS
from src.predict import predict_window

st.set_page_config(page_title="PAMAP2 HAR", page_icon="🏃", layout="wide")
cfg = load_config()
NAMES = {int(k): v for k, v in cfg["activities"].items()}


@st.cache_data
def load_signals():
    return pd.read_parquet(resolve(cfg["paths"]["interim"]))


@st.cache_data
def load_metrics():
    p = resolve(cfg["paths"]["metrics"])
    return json.loads(p.read_text()) if p.exists() else {}


st.title("🏃 Reconnaissance d'activité physique — PAMAP2")
st.caption("3 IMU (main, poitrine, cheville) + cardio | fenêtres 2,56 s | "
           "évaluation sur sujets jamais vus")

metrics = load_metrics()
tab1, tab2, tab3 = st.tabs(["Signaux", "Performance", "Reconnaître une fenêtre"])

with tab1:
    figdir = resolve(cfg["paths"]["figures"])
    f = figdir / "eda_signal_traces.png"
    if f.exists():
        st.image(str(f), width='stretch')
    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), ["eda_hr_by_activity.png", "eda_dom_freq.png"]):
        p = figdir / name
        if p.exists():
            col.image(str(p), width='stretch')

with tab2:
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-macro (sujets jamais vus)",
                  metrics["f1_macro_test_subject_wise"])
        c2.metric("Accuracy (test)", metrics["accuracy_test"])
        c3.metric("Split naïf (fuite !)", metrics["naive_split_f1_LEAKY"],
                  f"écart {metrics['leakage_gap']:+.3f}", delta_color="inverse")
        c4.metric("Champion déployé", metrics["champion_deployed"])
    figdir = resolve(cfg["paths"]["figures"])
    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), ["confusion_matrix.png",
                                    "leakage_comparison.png"]):
        p = figdir / name
        if p.exists():
            col.image(str(p), width='stretch')
    p = figdir / "feature_importance.png"
    if p.exists():
        st.image(str(p), width='stretch')

with tab3:
    st.write("Choisis un sujet **de test** et une activité : on extrait une "
             "fenêtre brute de 2,56 s et le modèle doit la reconnaître.")
    sig = load_signals()
    test_subjects = metrics.get("test_subjects",
                                sorted(sig["subject_id"].unique())[:3])
    c1, c2, c3 = st.columns(3)
    subject = c1.selectbox("Sujet (jamais vu à l'entraînement)", test_subjects)
    activity = c2.selectbox("Activité réelle", list(NAMES.keys()),
                            format_func=lambda a: NAMES[a])
    offset = c3.slider("Position dans l'enregistrement (fenêtre n°)", 0, 12, 3)

    if st.button("Reconnaître l'activité", type="primary"):
        block = sig[(sig["subject_id"] == subject) &
                    (sig["activity_id"] == activity)]
        win = cfg["segmentation"]["window_size"]
        w = block.iloc[offset * win:(offset + 1) * win]
        if len(w) < win:
            st.warning("Fenêtre incomplète, choisis une position plus petite.")
        else:
            channels = {c: w[c].tolist() for c in ACC_CHANNELS + GYRO_CHANNELS}
            res = predict_window(channels, w["heart_rate"].tolist())
            ok = res["activity_id"] == activity
            st.metric("Prédiction", res["activity_name"],
                      f"confiance {res['confidence']:.0%}")
            (st.success if ok else st.error)(
                f"{'✅ Correct' if ok else '❌ Erreur'} — vérité terrain : "
                f"{NAMES[activity]}")
            st.line_chart(w[["ankle_acc16_x"]].reset_index(drop=True))
            st.json(res["top3"])
