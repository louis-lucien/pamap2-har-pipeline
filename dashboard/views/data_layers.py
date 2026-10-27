import streamlit as st

import ui
from loaders import ACTIVITY_LABELS, LAYERS, cfg, column_profile, layer_stats, load_layer, raw_files

DOWNLOAD_MAX_ROWS = 20_000

ui.page_header("Données", "Catalogue des couches de données",
               "Chaque étape du pipeline produit une couche persistée. Comparez les volumes, "
               "inspectez le schéma et filtrez les enregistrements.")

if not raw_files():
    st.warning(f"Aucun fichier brut dans `{cfg['paths']['raw_dir']}` : relancez la génération "
               "ou le téléchargement des données.")
    st.stop()

stats = {name: layer_stats(name) for name in LAYERS}
ui.section("Lignage", "Volumétrie de chaque couche, de la source au jeu d'entraînement.")
ui.flow([(label, ui.fmt_int(stats[k]["rows"]),
          f"lignes · {stats[k]['cols']} colonnes · "
          + (f"{stats[k]['files']} fichiers · " if stats[k]["files"] > 1 else "")
          + f"{stats[k]['size_mb']:.1f} Mo {stats[k]['format']}")
         for k, (label, _) in LAYERS.items()])

ui.section("Couche à examiner")
c1, c2 = st.columns([2, 1])
with c1:
    layer = st.segmented_control("Couche", list(LAYERS), format_func=lambda k: LAYERS[k][0],
                                 default="raw", key="data_layer", label_visibility="collapsed") or "raw"
subject_file = None
if layer == "raw":
    subject_file = c2.selectbox("Fichier", raw_files(), key="raw_file", label_visibility="collapsed")
data = load_layer(layer, subject_file)
shown_path = f"{cfg['paths']['raw_dir']}/{subject_file}" if subject_file else cfg["paths"][layer]
st.caption(f"{LAYERS[layer][1]}  ·  `{shown_path}`")

overview, schema, explorer = st.tabs(["Aperçu", "Schéma", "Explorateur"])

with overview:
    ui.kpi_row([
        ("Lignes", ui.fmt_int(len(data)),
         "fenêtres de 2,56 s" if layer == "features" else "mesures à 100 Hz", True),
        ("Colonnes", data.shape[1]),
        ("Sujets", data["subject_id"].nunique() if "subject_id" in data else "—", "dans cette vue"),
        ("Cellules vides", f"{data.isna().mean().mean():.1%}", "valeurs manquantes"),
    ])
    st.write("")
    st.markdown("**Échantillon (10 premières lignes)**")
    st.dataframe(data.head(10), hide_index=True, width="stretch")
    st.markdown("**Statistiques descriptives (colonnes numériques)**")
    st.dataframe(data.describe(include="number").T.round(2), width="stretch")

with schema:
    st.dataframe(column_profile(layer, subject_file), hide_index=True, width="stretch",
                 column_config={"Manquants (%)": st.column_config.ProgressColumn(
                     "Manquants (%)", min_value=0, max_value=100, format="%.1f %%")})

with explorer:
    view = data
    with st.container(border=True):
        f1, f2 = st.columns(2)
        if layer != "raw" and "subject_id" in data:
            subjects = f1.multiselect("Sujets", sorted(data["subject_id"].unique()), key=f"subj_{layer}")
            if subjects:
                view = view[view["subject_id"].isin(subjects)]
        if "activity_id" in data:
            activities = f2.multiselect("Activités", sorted(data["activity_id"].unique()),
                                        format_func=lambda a: ACTIVITY_LABELS.get(int(a), str(a)),
                                        key=f"act_{layer}")
            if activities:
                view = view[view["activity_id"].isin(activities)]
    st.caption(f"{ui.fmt_int(len(view))} lignes correspondent aux filtres · 500 premières affichées")
    st.dataframe(view.head(500), hide_index=True, width="stretch")
    st.download_button(f"Exporter la sélection (CSV, {ui.fmt_int(DOWNLOAD_MAX_ROWS)} lignes max)",
                       view.head(DOWNLOAD_MAX_ROWS).to_csv(index=False).encode("utf-8"),
                       file_name=f"{layer}_selection.csv", mime="text/csv", icon=":material/download:")
