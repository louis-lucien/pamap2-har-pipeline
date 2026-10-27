import streamlit as st

import ui
from loaders import cfg

seg = cfg["segmentation"]

ui.page_header("Documentation", "Méthodologie",
               "Choix de conception du pipeline et garde-fous contre les biais d'évaluation.")

ui.section("Données")
st.markdown(f"""
Jeu au format **PAMAP2** (UCI) : 3 centrales inertielles (poignet, poitrine, cheville) échantillonnées à
**{cfg['sensors']['sampling_rate_hz']} Hz** et un cardiofréquencemètre (~{cfg['sensors']['heart_rate_hz']} Hz).
Chaque fichier contient 54 colonnes : horodatage, activité, fréquence cardiaque, puis 17 mesures par IMU.
""")

ui.section("Prétraitement")
st.markdown(f"""
- Suppression des périodes de **transition** (activité 0).
- Suppression des colonnes d'**orientation**, déclarées invalides par la documentation officielle.
- Interpolation des trous IMU jusqu'à **{cfg['cleaning']['max_gap_interpolate']} échantillons**, sujet par sujet.
""")

ui.section("Segmentation et variables")
st.markdown(f"""
- Fenêtres glissantes de **{seg['window_size']} échantillons** ({seg['window_size'] / cfg['sensors']['sampling_rate_hz']:.2f} s),
  recouvrement **{seg['overlap']:.0%}** ; rejet si plus de {1 - seg['min_valid_ratio']:.0%} de valeurs manquantes.
- Par canal accéléromètre / gyroscope : moyenne, écart-type, min, max, médiane, énergie.
- Par emplacement (poignet, poitrine, cheville) : **fréquence dominante** de l'accélération (FFT).
- Rythme cardiaque : moyenne et écart-type sur la fenêtre.
""")

ui.section("Protocole d'évaluation")
st.markdown("""
- **Split par sujet** : les sujets de test sont exclus de l'entraînement. Un split aléatoire mélangerait des
  fenêtres d'une même personne entre entraînement et test et surestimerait la performance (mesuré dans la
  page Performance).
- Sélection du champion par **validation croisée groupée** (GroupShuffleSplit sur le sujet).
- Comparaison systématique à une **baseline** (régression logistique).
- Métrique principale : **F1-macro**, qui pondère toutes les activités à égalité.
""")

ui.section("Architecture technique")
st.markdown("""
| Brique | Implémentation |
|---|---|
| Ingestion | `src/ingestion.py` : acquisition (UCI ou local), schéma des 54 colonnes |
| Nettoyage | `src/cleaning.py` : transitions, orientation, interpolation, sortie Parquet |
| Features | `src/features.py` : fenêtrage et extraction temporelle / fréquentielle |
| Entraînement | `src/train.py` : baseline vs challenger, CV groupée |
| Évaluation | `src/evaluate.py` : métriques, démonstration de fuite, figures |
| Orchestration | `pipelines/flow.py`, `run_pipeline.py` |
| Service | `api/main.py` (API REST), `dashboard/` (cette application) |
| Configuration | `config/config.yaml` |
""")

ui.section("Limites")
st.markdown("""
- Jeu de démonstration synthétique conforme au format PAMAP2 ; nombre de sujets réduit.
- Huit activités seulement : les activités optionnelles du protocole ne sont pas couvertes.
- Pas de lissage temporel des prédictions entre fenêtres successives.
""")
