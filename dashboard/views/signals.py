import streamlit as st

import ui
from loaders import ACTIVITIES, figure_path, load_layer

features = load_layer("features")

ui.page_header("Données", "Analyse des signaux",
               "Signature de chaque activité dans les capteurs inertiels et le rythme cardiaque.")

ui.kpi_row([
    ("Fenêtres", ui.fmt_int(len(features)), "2,56 s chacune", True),
    ("Sujets", features["subject_id"].nunique()),
    ("Activités", features["activity_id"].nunique()),
    ("FC moyenne", f"{features['hr_mean'].mean():.0f} bpm", "toutes activités"),
])

ui.section("Traces brutes", "Accélération et rythme cardiaque sur un extrait de chaque activité.")
ui.figure(figure_path("eda_signal_traces.png"), "Extraits de signaux par activité")

ui.section("Variables discriminantes")
c1, c2 = st.columns(2)
with c1:
    ui.figure(figure_path("eda_hr_by_activity.png"), "Rythme cardiaque par activité")
with c2:
    ui.figure(figure_path("eda_dom_freq.png"), "Fréquence dominante (FFT) par activité")
ui.note("La fréquence dominante donne la cadence du mouvement (≈ 1,9 Hz en marche, ≈ 2,9 Hz en course) : "
        "c'est elle qui sépare des activités aux amplitudes proches.")

ui.section("Équilibre des classes")
counts = (features.groupby("activity_id").size().rename(index=ACTIVITIES)
          .rename("Fenêtres").to_frame())
counts["Part"] = (counts["Fenêtres"] / counts["Fenêtres"].sum() * 100).round(1)
st.dataframe(counts, width="stretch",
             column_config={"Part": st.column_config.ProgressColumn("Part (%)", min_value=0, max_value=100,
                                                                    format="%.1f %%")})
