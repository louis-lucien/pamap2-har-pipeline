# 🏃 PAMAP2 — Reconnaissance d'Activité Physique (HAR)

Pipeline **de bout en bout** qui, à partir des signaux bruts de 3 centrales
inertielles (main, poitrine, cheville) et d'un cardiofréquencemètre, reconnaît
**quelle activité physique** une personne est en train de faire — et l'expose
en temps réel via une **API** et un **dashboard**.

> Dataset de référence : [PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
> (UCI ML Repository — 9 sujets, 3 IMU à 100 Hz + cardio ~9 Hz, fichiers .dat).

---

## 🎯 Ce que fait ce projet

```
.dat bruts ──► Ingestion ──► Nettoyage ──► Fenêtrage ──► Features ──► Champion ──► API + Dashboard
 (9 sujets)    (54 col.,     (interpolation (2,56 s,     (temporel +   (baseline vs  (fenêtre brute →
               contrôle      par sujet,     50% overlap)  fréquentiel   RF, le        activité reconnue)
               qualité)      transitions)                 FFT)          meilleur gagne)
```

**Le point méthodologique central : la séparation PAR SUJET.** Deux fenêtres
du même sujet se ressemblent trop ; un split naïf donne un score parfait
mais illusoire (fuite). On évalue sur des **personnes jamais vues** — et le
pipeline **démontre l'écart chiffré** entre les deux protocoles.

---

## 📁 Structure

```
pamap2-har-pipeline/
├── config/config.yaml          # tous les paramètres (fenêtre, activités, split)
├── data/{raw/Protocol,interim,processed}
├── src/
│   ├── config.py               # config + logging
│   ├── ingestion.py            # E : téléchargement UCI + parsing .dat 54 col.
│   ├── cleaning.py             # T : transitions, orientation, interpolation
│   ├── features.py             # fenêtrage + features temps/fréquence (FFT)
│   ├── eda.py                  # signaux, cardio, cadence par activité
│   ├── train.py                # démo fuite + baseline + sélection champion
│   ├── evaluate.py             # métriques multi-classes + figures
│   ├── predict.py              # inférence sur FENÊTRE BRUTE (même code features)
│   └── generate_sample_data.py # jeu synthétique fidèle (format .dat officiel)
├── pipelines/flow.py           # orchestration Prefect (6 tâches)
├── api/main.py                 # API temps réel (fenêtre brute -> activité)
├── dashboard/app.py            # Streamlit : signaux, perfs, démo live
├── tests/test_pipeline.py      # 4 tests (dont anti-fuite)
├── run_pipeline.py             # ▶ point d'entrée principal
└── requirements.txt
```

---

## 🚀 Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Données : téléchargement UCI automatique au premier run (mode par défaut).
# Sans internet : python -m src.generate_sample_data  (jeu de démo fidèle)

python run_pipeline.py                 # tout le pipeline
uvicorn api.main:app --reload          # API  -> http://127.0.0.1:8000/docs
streamlit run dashboard/app.py         # dashboard (démo de reconnaissance live)
pytest -q                              # tests
python -m pipelines.flow               # version orchestrée Prefect
```

---

## 📊 Résultats (jeu de démonstration)

| Protocole d'évaluation | F1-macro | Lecture |
|---|---|---|
| Split naïf (fenêtres au hasard) | **1.000** | ⚠️ Illusion — fuite de données |
| **Split par sujet (honnête)** | **0.955** | Vraie généralisation à des inconnus |
| Baseline LogReg (CV groupée) | 0.977 | A battu le Random Forest (0.770) |

**Champion déployé : la régression logistique** — sélectionnée automatiquement
car elle généralise mieux inter-sujets que le Random Forest, qui sur-apprenait
les particularités des sujets d'entraînement. La complexité doit se mériter.

Les confusions restantes concernent marche / montée / descente d'escaliers —
exactement les classes difficiles de la littérature HAR.

---

## 🧠 Choix méthodologiques clés

1. **Split par sujet, prouvé** : l'écart naïf (1.000) vs honnête (0.955) est
   calculé et tracé (`reports/figures/leakage_comparison.png`).
2. **Une seule fonction de features** partagée entre l'entraînement et l'API :
   zéro *training-serving skew*.
3. **Fenêtres jamais à cheval** sur deux sujets ou deux activités.
4. **Sélection de champion** : baseline et challenger comparés en CV groupée,
   le meilleur est déployé — décision d'ingénieur, pas dogme du modèle complexe.
5. **Interpolation par sujet** (jamais à travers deux personnes) et rejet des
   fenêtres trop incomplètes plutôt que suppression aveugle de lignes.

Voir `GUIDE_PEDAGOGIQUE.md` pour l'explication complète pas à pas.
