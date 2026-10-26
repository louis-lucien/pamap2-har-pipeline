# 📘 Guide pédagogique complet — Reconnaissance d'Activité Physique (PAMAP2)

**Projet 2 — Prédiction de l'activité physique à partir de capteurs corporels**

Ce document explique tout ce qui a été fait, dans l'ordre, avec des mots
simples et des analogies. Objectif : qu'un novice comprenne le raisonnement du
début à la fin, et que tu puisses défendre chaque choix devant un jury.

---

## Table de traçabilité : chaque section ↔ le code qui l'implémente

| Section du guide | Fichier(s) | Vérifiable par |
|---|---|---|
| 3. Comprendre les capteurs | `src/ingestion.py` (54 colonnes) | rapport qualité dans les logs |
| 4. Ingestion | `src/ingestion.py` (`acquire`+`extract`) | téléchargement UCI + repli local |
| 5. Nettoyage signal | `src/cleaning.py` | filtres comptés, `data/interim/*.parquet` |
| 6. Fenêtrage | `src/features.py` (`build_features`) | 2 448 fenêtres, 0 à cheval (testé) |
| 7. Extraction de features | `src/features.py` (`extract_features_from_window`) | 119 features, FFT incluse |
| 8. Le piège : split par sujet | `src/train.py` | écart chiffré dans `metrics.json` + figure |
| 9. Baseline + champion | `src/train.py` | champion loggé et sauvegardé |
| 10. Évaluation | `src/evaluate.py` | matrice 8×8, `reports/metrics.json` |
| 11. API + Dashboard | `api/main.py`, `dashboard/app.py` | `/predict` testé, démo live |
| 12. MLOps | `config/`, `pipelines/flow.py`, `tests/` | 4 tests pytest dont un anti-fuite |

---

## 1. Le problème en une phrase

> **À partir des signaux de 3 capteurs portés sur le corps (main, poitrine,
> cheville) et du rythme cardiaque, deviner automatiquement ce que la personne
> est en train de faire : marcher, courir, pédaler, monter des escaliers…**

C'est la **reconnaissance d'activité humaine** (HAR — *Human Activity
Recognition*), la technologie derrière ta montre connectée qui sait toute
seule que tu as commencé à courir.

**Analogie.** Ferme les yeux et pose ta main sur l'épaule d'un ami : rien
qu'aux secousses, tu distingues s'il marche, court ou est assis. Nos capteurs
sont cette main posée à trois endroits du corps, 100 fois par seconde.

**Applications réelles** : suivi médical de patients, détection de chute chez
les personnes âgées, coaching sportif, ergonomie au travail. Et pour toi, Era,
c'est méthodologiquement le même problème que superviser un onduleur SMA avec
des registres Modbus : des séries temporelles multi-capteurs à des fréquences
différentes, dont il faut extraire un état.

---

## 2. Vue d'ensemble : la chaîne complète

```
 ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐
 │ .dat bruts│─▶│ Ingestion │─▶│Nettoyage │─▶│ Fenêtrage │─▶│ Champion │─▶│API + Dash │
 │ 9 sujets  │  │ 54 col.   │  │ signal   │  │ +features │  │ (LogReg  │  │ temps réel│
 │ 100 Hz    │  │           │  │          │  │ FFT       │  │ vs RF)   │  │           │
 └───────────┘  └───────────┘  └──────────┘  └───────────┘  └──────────┘  └───────────┘
   data/raw       contrôle       Parquet       Parquet        .joblib       fenêtre brute
   Protocol/      qualité        interim       processed      +métriques    -> activité
```

Même architecture en couches que le projet churn (bronze/argent/or), mêmes
quatre piliers (Data Engineering, Data Science, MLOps, Déploiement). Ce qui
change : la **nature de la donnée** (du signal, pas des tables) et le
**piège méthodologique** (le sujet, pas le temps).

---

## 3. Étape 1 — Comprendre les capteurs et la donnée

Chaque sujet porte :
- **3 IMU** (centrales inertielles) à **100 Hz** : une au poignet dominant,
  une sur la poitrine, une à la cheville. Chacune mesure l'accélération 3D
  (deux gammes : ±16g et ±6g), la rotation 3D (gyroscope), le champ magnétique
  3D (magnétomètre) et la température.
- **1 cardiofréquencemètre** à **~9 Hz**.

Une ligne du fichier `.dat` = un instant. **54 colonnes** : horodatage,
étiquette d'activité, cardio, puis 3 × 17 colonnes d'IMU.

Trois pièges documentés dans la spécification officielle, que notre code gère :

| Piège | Pourquoi | Notre traitement |
|---|---|---|
| Cardio à 9 Hz vs IMU à 100 Hz | ~91 % de NaN sur la colonne cardio, **par construction** | interpolation linéaire par sujet |
| 4 colonnes « orientation » par IMU | déclarées **invalides** par les auteurs | supprimées |
| `activityID = 0` | périodes transitoires entre activités | supprimées |
| Accéléromètre ±6g | **sature** pendant la course | on utilise le ±16g |

**Analogie 100 Hz.** 100 mesures par seconde, c'est comme filmer le mouvement
avec une caméra à 100 images/seconde : assez rapide pour capter chaque détail
d'un pas (un pas de marche dure ~0,5 s, soit 50 images).

> 🔧 **Dans le projet** : l'acquisition est automatique (`src/ingestion.py`
> télécharge le zip UCI officiel, avec repli local). Faute d'internet, le
> générateur `src/generate_sample_data.py` produit 9 sujets au **format .dat
> exact** (54 colonnes, NaN cardio à 9 Hz, pertes sans fil, transitions), avec
> une physique plausible : gravité selon la posture, composantes périodiques
> pour la marche/course/vélo, cardio proportionnel à l'effort, et une
> **variabilité entre sujets** (chacun a sa démarche) — indispensable pour que
> la leçon du split par sujet soit réaliste.

---

## 4. Étape 2 — Ingestion (Extract)

**Fichier : `src/ingestion.py`**

Lire les 9 fichiers `.dat` (352 800 lignes), nommer les 54 colonnes selon la
spécification, ajouter la colonne `subject_id`, et produire le rapport
qualité : combien de NaN cardio (attendu ~91 %), combien de NaN IMU (pertes
sans fil, ~0,4 %), combien de transitoire (~8 %). On **diagnostique avant de
soigner** — comme dans le projet churn.

---

## 5. Étape 3 — Nettoyage (Transform), version « signal »

**Fichier : `src/cleaning.py`**

**La grande différence avec des données tabulaires** : on ne peut pas
supprimer les lignes NaN n'importe comment, car cela **casserait la continuité
temporelle**. Le fenêtrage (étape suivante) suppose des échantillons contigus
à 100 Hz : enlever une ligne au milieu, c'est comme couper une image au milieu
d'un film — tout ce qui suit est décalé.

Donc on **interpole** les petits trous (≤ 0,2 s) au lieu de supprimer, on
interpole **par sujet** (jamais entre la fin du sujet 103 et le début du 104 !),
et les trous trop longs restent NaN : c'est la segmentation qui rejettera les
fenêtres trop incomplètes (`min_valid_ratio` dans la config).

Sur notre jeu : 352 800 → 324 000 lignes (91,8 % conservées, l'essentiel du
retrait = les transitions), NaN résiduels 2,68 % → 0,00 %.

---

## 6. Étape 4 — Le fenêtrage : l'analogie Shazam

**Fichier : `src/features.py`**

Un échantillon isolé d'accéléromètre ne dit **rien** : `ankle_x = 3,2 m/s²`
peut appartenir à la marche, à la course ou au vélo. Ce qui identifie une
activité, c'est la **forme du signal dans le temps** — sa signature.

**Analogie Shazam.** Shazam ne reconnaît pas une chanson à partir d'une seule
note, mais d'un **extrait de ~3 secondes** dont il calcule l'empreinte. Nous
faisons pareil : on découpe le signal en **fenêtres de 2,56 s** (256
échantillons à 100 Hz — valeur standard de la littérature HAR) et chaque
fenêtre devient un « extrait » à classifier.

Trois règles d'or du fenêtrage, toutes implémentées et testées :

1. **Recouvrement de 50 %** : les fenêtres se chevauchent pour ne pas rater
   une signature qui tomberait entre deux fenêtres (comme des tuiles qui se
   recouvrent pour ne pas laisser passer la pluie).
2. **Jamais à cheval** sur deux activités : une fenêtre moitié-marche
   moitié-course aurait une étiquette absurde. On segmente à l'intérieur des
   blocs contigus (même sujet, même activité).
3. **Rejet des fenêtres trop trouées** : > 20 % de NaN restants → rejetée.

Résultat : **2 448 fenêtres**, parfaitement équilibrées (306 par activité).

---

## 7. Étape 5 — Extraction de caractéristiques (temps + fréquence)

**Fichier : `src/features.py` — fonction `extract_features_from_window`**

Chaque fenêtre (256 lignes × 18 canaux) est résumée en **119 nombres** :

| Famille | Contenu | Ce que ça capture |
|---|---|---|
| **Temporel** (par canal) | mean, std, min, max, médiane, énergie | intensité et variabilité du mouvement |
| **Magnitude** ‖a‖ par IMU | mean, std de √(x²+y²+z²) | mouvement **indépendant de l'orientation** du capteur |
| **Fréquentiel** (FFT) | fréquence dominante par IMU | **la cadence** : ~1,9 Hz en marche, ~2,9 Hz en course |
| **Physiologique** | cardio moyen, écart-type | l'effort (155 bpm ≠ 72 bpm) |

**La FFT expliquée simplement.** La transformée de Fourier décompose un signal
en ses rythmes, comme une oreille de percussionniste qui isole le tempo dans
un sabar : « ce morceau est à 120 battements par minute ». Pour la cheville
d'un marcheur, la FFT répond « le rythme dominant est 1,9 pas par seconde » —
et c'est LA caractéristique qui sépare marche et course, deux activités aux
allures sinon similaires. La figure `eda_dom_freq.png` le montre visuellement.

**Point d'architecture crucial** : cette fonction est **la même** en
entraînement et dans l'API. Si l'API recalculait ses features avec un autre
code, la moindre divergence (une normalisation oubliée, une bande de
fréquence différente) fausserait silencieusement les prédictions en
production. C'est un défaut classique appelé *training-serving skew* — ici,
il est **structurellement impossible**.

---

## 8. Étape 6 — LE piège : la séparation par sujet

**C'est le point central du projet, l'équivalent du fenêtrage temporel du
projet churn. À maîtriser parfaitement pour la soutenance.**

### Le piège

Deux fenêtres consécutives du même sujet se ressemblent énormément : même
démarche, même morphologie, même capteur posé au même endroit — et 50 % de
recouvrement en plus. Si on coupe train/test **au hasard sur les fenêtres**,
le modèle retrouve dans le test des fenêtres quasi identiques à celles du
train. Il « reconnaît la personne » au lieu de « reconnaître l'activité ».

Notre pipeline le **prouve par le chiffre** : split naïf → **F1 = 1,000**.
Un score parfait. Trop parfait. C'est le symptôme, pas la réussite.

### La bonne question

En production, la montre connectée sera portée par des gens que le modèle n'a
**jamais vus**. La bonne question est donc : « le modèle marche-t-il sur une
personne inconnue ? » On sépare **par sujet** : 6 sujets pour apprendre,
3 sujets (102, 106, 108) réservés au test, qui n'ont contribué à **aucun**
apprentissage. C'est le protocole *leave-subjects-out* de la littérature HAR.

Résultat honnête : **F1 = 0,955**. L'écart de 0,045 avec le score naïf, c'est
la **quantité de fuite** — calculée, journalisée dans `metrics.json`, et
tracée dans `leakage_comparison.png`. (Sur le vrai PAMAP2, où les humains
sont bien plus variés que notre simulation, l'écart est typiquement bien plus
grand : naïf ~0,99 contre honnête ~0,70–0,85.)

**Analogie du médecin.** Un étudiant en médecine qui révise sur les dossiers
des patients A, B, C et qu'on évalue… sur d'autres pages des mêmes dossiers
aura l'air brillant : il reconnaît les patients, pas les maladies. Le vrai
examen, c'est un patient inconnu. Nos 3 sujets de test sont ces patients
inconnus.

**Le parallèle avec le projet churn** — à faire remarquer au jury : dans les
deux projets, la fuite vient d'une **corrélation cachée entre train et test**
(le temps pour le churn, l'identité du sujet ici). La compétence transférable,
c'est le réflexe de se demander : *« en production, qu'est-ce que le modèle ne
connaîtra vraiment pas ? »* — et de construire l'évaluation là-dessus.

Même la validation croisée pour les hyperparamètres est **groupée par sujet**
(`GroupShuffleSplit`) : la rigueur s'applique partout, pas seulement au test
final. Et un test pytest (`test_subject_wise_split_no_overlap`) garantit
qu'aucun sujet de test ne fuit dans le train.

---

## 9. Étape 7 — Baseline et sélection du champion

**Fichier : `src/train.py`**

Comme pour le churn : régression logistique en **baseline**, Random Forest en
challenger, comparés à la loyale (même split honnête, même CV groupée).

Résultat inattendu et instructif : **la baseline gagne** (F1 CV groupée 0,977
contre 0,770 pour le Random Forest). Pourquoi ? Le Random Forest, plus
flexible, sur-apprend les particularités des 6 sujets d'entraînement ; la
régression logistique, plus contrainte, capte les régularités **communes à
tous les humains** et généralise mieux aux sujets inconnus.

Le pipeline en tire la conséquence d'ingénieur : **sélection automatique du
champion** — le modèle déployé est celui qui gagne en CV groupée, quel qu'il
soit. Ici c'est la LogReg : plus légère, plus interprétable, meilleure. La
complexité doit se mériter ; un modèle plus gros n'est pas un meilleur modèle.

---

## 10. Étape 8 — Évaluation multi-classes

**Fichier : `src/evaluate.py`**

8 classes → les métriques changent par rapport au churn binaire :

| Métrique | Ce qu'elle dit |
|---|---|
| **Accuracy** | proportion de fenêtres bien classées (fiable ici car classes équilibrées) |
| **F1-macro** | moyenne des F1 par classe, chaque activité compte pareil — la course ne peut pas « cacher » les escaliers |
| **Matrice de confusion 8×8** | OÙ le modèle se trompe |

La matrice révèle que les erreurs restantes concernent **marche / montée /
descente d'escaliers** — les postures (allongé, assis, debout) et les
activités à signature forte (course, vélo) sont parfaites. C'est exactement le
schéma de la littérature HAR : les trois formes de locomotion à ~1,5–1,9 Hz se
ressemblent. En soutenance, savoir dire *où* et *pourquoi* ton modèle se
trompe vaut autant que le score.

---

## 11. Étape 9 — Servir le modèle : la vraie prédiction temps réel

**Fichiers : `api/main.py`, `dashboard/app.py`**

L'énoncé demande « la prédiction en temps réel à partir de nouvelles
données ». Notre API prend ça au sérieux : `POST /predict` reçoit une
**fenêtre brute** (256 échantillons × 18 canaux + cardio optionnel) —
exactement ce qu'enverrait un smartphone ou un bracelet — calcule les features
côté serveur **avec la fonction de l'entraînement**, et renvoie l'activité,
la confiance, et le top 3 des probabilités.

Le dashboard a un onglet démo : tu choisis un **sujet de test** (jamais vu à
l'entraînement !), une activité, une position dans l'enregistrement — et le
modèle doit reconnaître la fenêtre en direct, avec le signal affiché et le
verdict correct/incorrect. C'est la démonstration la plus convaincante
possible en soutenance.

---

## 12. MLOps — identique au projet churn, adapté au signal

Config centralisée (`config.yaml` : taille de fenêtre, recouvrement,
activités, ratio de sujets test — tout est réglable sans toucher au code),
orchestration Prefect à 6 tâches avec retries sur le téléchargement, logs
horodatés, reproductibilité (`random_state=42`), et **4 tests pytest** dont
deux spécifiques au domaine : aucune fenêtre à cheval, et aucun sujet de test
dans le train (test anti-fuite).

---

## 13. Les résultats et comment les lire

| Mesure | Valeur | Lecture |
|---|---|---|
| Split naïf (fenêtres au hasard) | **1,000** | ⚠️ le mensonge — fuite démontrée |
| **F1-macro, sujets jamais vus** | **0,955** | la vraie performance |
| Accuracy test | 0,956 | cohérente (classes équilibrées) |
| Baseline LogReg (CV groupée) | 0,977 | bat le RF (0,770) → **champion** |

> ⚠️ **Honnêteté intellectuelle** : chiffres obtenus sur le jeu synthétique.
> Sur le vrai PAMAP2 (humains réellement variés, capteurs qui glissent,
> activités mal exécutées), attends-toi à un F1 sujet-wise de l'ordre de
> **0,70–0,85** — et un écart de fuite bien plus spectaculaire. La méthode est
> identique ; les chiffres réels seront tes chiffres.

---

## 14. Comment défendre ce projet devant un jury

**Sur la valeur**
> « C'est la technologie des montres connectées et du suivi médical à
> distance : reconnaître l'activité sans intervention de l'utilisateur. »

**Sur la rigueur (le point fort)**
> « J'ai démontré chiffres à l'appui que le protocole d'évaluation naïf donne
> un score parfait mais illusoire, par fuite inter-sujets. Mon évaluation est
> faite sur trois personnes que le modèle n'a jamais vues, avec une validation
> croisée elle-même groupée par sujet. »

**Sur le traitement du signal**
> « Fenêtrage 2,56 s à 50 % de recouvrement, features temporelles et
> fréquentielles — la FFT extrait la cadence de pas, qui sépare marche et
> course. Les fenêtres ne sont jamais à cheval sur deux activités. »

**Sur l'ingénierie**
> « La fonction d'extraction de features est partagée entre l'entraînement et
> l'API : le training-serving skew est structurellement impossible. Et le
> pipeline sélectionne automatiquement le champion — ici la régression
> logistique a battu le Random Forest en généralisation inter-sujets, donc
> c'est elle qui est déployée. »

**Questions pièges probables**

- *« Pourquoi 256 échantillons et 50 % de recouvrement ? »* → Standard de la
  littérature HAR : 2,56 s couvrent plusieurs cycles de pas (un pas ~0,5 s) ;
  le recouvrement évite de rater une signature entre deux fenêtres. Et ce sont
  des paramètres de config, pas des constantes magiques — testables.
- *« Pourquoi la LogReg bat le Random Forest ? »* → Le RF sur-apprend les
  particularités des 6 sujets d'entraînement ; la LogReg capte les régularités
  communes aux humains. Sur le vrai jeu, plus varié, le verdict peut
  s'inverser — c'est pour ça que la sélection du champion est automatique.
- *« Et le déséquilibre des classes ? »* → Équilibré ici par construction ; sur
  le vrai jeu, certaines activités sont plus courtes → `class_weight=balanced`
  est déjà en place, et le F1-macro (pas l'accuracy) est la métrique pilote.
- *« Un seul split par sujet, c'est fragile, non ? »* → Exact, avec 9 sujets la
  variance est réelle. L'extension rigoureuse est le **leave-one-subject-out**
  (9 entraînements, chaque sujet servant une fois de test) — mentionner que
  c'est la suite naturelle montre que tu connais les limites de ton protocole.

---

## 15. Glossaire

| Terme | Définition simple |
|---|---|
| **HAR** | Human Activity Recognition — reconnaître l'activité d'une personne via capteurs |
| **IMU** | Centrale inertielle : accéléromètre + gyroscope + magnétomètre |
| **Hz (Hertz)** | Mesures par seconde. 100 Hz = 100 mesures/s |
| **Fenêtrage** | Découper le signal en tranches de durée fixe à classifier |
| **Recouvrement** | Chevauchement entre fenêtres consécutives (50 % ici) |
| **FFT** | Transformée de Fourier : décompose un signal en ses rythmes |
| **Fréquence dominante** | Le rythme le plus fort du signal (la cadence de pas) |
| **F1-macro** | Moyenne des F1 par classe — chaque activité pèse pareil |
| **Fuite inter-sujets** | Le test contient des fenêtres « jumelles » du train |
| **Leave-subjects-out** | Évaluer sur des personnes absentes de l'entraînement |
| **GroupShuffleSplit** | Split scikit-learn qui respecte les groupes (sujets) |
| **Training-serving skew** | Écart entre les features du labo et celles de la prod |
| **Champion/challenger** | Comparer les modèles et déployer le vainqueur |

---

*Fin du guide. Pour les commandes pratiques, voir `README.md`.*
