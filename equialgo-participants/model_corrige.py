import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# CONFIG
# ============================================================

TRAIN = "data/donnees_demandes.csv"
TEST = "data/candidats_evaluation.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

y = train["decision_octroi"].astype(int)

X = train.drop(columns=["decision_octroi"]).copy()
T = test.copy()


# ============================================================
# GROUPES
# ============================================================

REMOTE = {
    "Bas-Saint-Laurent",
    "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine"
}

for df in [X, T]:

    df["groupe"] = np.where(
        df["region_administrative"].isin(REMOTE),
        "eloignee",
        "centre"
    )


# ============================================================
# 1. SCORE DE MÉRITE
# ============================================================

def percentile(x):
    return pd.Series(x).rank(pct=True).values


def build_merit_score(df):

    cote = percentile(df["cote_r_equivalent"])

    # Plus bas revenu = davantage de besoin
    revenu = 1 - percentile(
        df["revenu_familial_estime"]
    )

    # Plus d'heures = contrainte plus importante
    heures = percentile(
        df["heures_travail_semaine"]
    )

    # Plus loin = contrainte géographique
    distance = percentile(
        df["distance_domicile_campus_km"]
    )

    premiere_gen = (
        df["premiere_generation_universitaire"]
        .astype(float)
    )

    # --------------------------------------------------------
    # SCORE PRINCIPAL
    #
    # Cote R = signal dominant.
    # Les autres variables représentent surtout le besoin.
    # --------------------------------------------------------

    score = (
        0.55 * cote
        + 0.15 * revenu
        + 0.12 * heures
        + 0.10 * distance
        + 0.08 * premiere_gen
    )

    return score


X["score_merite"] = build_merit_score(X)
T["score_merite"] = build_merit_score(T)


# ============================================================
# 2. SCORE DE MÉRITE PUR
# ============================================================

X["score_academique"] = percentile(
    X["cote_r_equivalent"]
)

T["score_academique"] = percentile(
    T["cote_r_equivalent"]
)


# ============================================================
# 3. SCORE DE BESOIN
# ============================================================

def build_need_score(df):

    revenu = 1 - percentile(
        df["revenu_familial_estime"]
    )

    heures = percentile(
        df["heures_travail_semaine"]
    )

    distance = percentile(
        df["distance_domicile_campus_km"]
    )

    premiere_gen = (
        df["premiere_generation_universitaire"]
        .astype(float)
    )

    return (
        0.40 * revenu
        + 0.25 * heures
        + 0.20 * distance
        + 0.15 * premiere_gen
    )


X["score_besoin"] = build_need_score(X)
T["score_besoin"] = build_need_score(T)


# ============================================================
# 4. MODÈLE HISTORIQUE SANS VARIABLES RÉGIONALES
# ============================================================

features_merit = [
    "cote_r_equivalent",
    "revenu_familial_estime",
    "heures_travail_semaine",
    "distance_domicile_campus_km",
    "premiere_generation_universitaire",
    "score_merite",
    "score_academique",
    "score_besoin"
]

X_model = X[features_merit].copy()
T_model = T[features_merit].copy()


# ============================================================
# 5. MODÈLE FLEXIBLE
# ============================================================

model = ExtraTreesClassifier(
    n_estimators=2500,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features=0.9,
    criterion="entropy",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. OOF
# ============================================================

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))

for fold, (tr, va) in enumerate(
    skf.split(X_model, y), 1
):

    print(f"Fold {fold}/10")

    model.fit(
        X_model.iloc[tr],
        y.iloc[tr]
    )

    oof[va] = model.predict_proba(
        X_model.iloc[va]
    )[:, 1]


# ============================================================
# 7. IMPORTANT :
#    ON NE CHERCHE PAS SEULEMENT LE SEUIL 0.5
# ============================================================

print("\n" + "=" * 70)
print("MODÈLE SANS RÉGION")
print("=" * 70)

for threshold in np.arange(
    0.30,
    0.71,
    0.01
):

    pred = (
        oof >= threshold
    ).astype(int)

    acc = accuracy_score(
        y,
        pred
    )

    if acc > 0.90:

        print(
            f"seuil={threshold:.2f} "
            f"accuracy={acc:.4f}"
        )


# ============================================================
# 8. APPRENTISSAGE FINAL
# ============================================================

model.fit(
    X_model,
    y
)

test_probability = model.predict_proba(
    T_model
)[:, 1]


# ============================================================
# 9. COMBINAISON MODÈLE + SCORE LATENT
# ============================================================

merit = T["score_merite"].values
academic = T["score_academique"].values
need = T["score_besoin"].values


# Plusieurs conceptions possibles
scores = {

    "modele_pur":
        test_probability,

    "merite_60":
        0.60 * test_probability
        + 0.40 * merit,

    "merite_70":
        0.70 * test_probability
        + 0.30 * merit,

    "academique_50":
        0.50 * test_probability
        + 0.30 * academic
        + 0.20 * need,

    "merite_besoin":
        0.50 * test_probability
        + 0.30 * merit
        + 0.20 * need,

    "equilibre":
        0.40 * test_probability
        + 0.35 * academic
        + 0.25 * need
}


# ============================================================
# 10. CORRECTION RÉGIONALE PAR RANG
# ============================================================

def group_rank_adjustment(
    score,
    groups,
    alpha
):

    score = np.asarray(score)

    global_rank = (
        pd.Series(score)
        .rank(pct=True)
        .values
    )

    group_rank = np.zeros(len(score))

    for g in np.unique(groups):

        idx = np.where(groups == g)[0]

        group_rank[idx] = (
            pd.Series(score[idx])
            .rank(pct=True)
            .values
        )

    # alpha = 0 → classement global
    # alpha = 1 → classement entièrement intra-groupe

    return (
        (1 - alpha) * global_rank
        + alpha * group_rank
    )


# ============================================================
# 11. GÉNÉRATION DE PLUSIEURS CANDIDATS
# ============================================================

groups = T["groupe"].values

candidates = []


for score_name, base_score in scores.items():

    for alpha in np.arange(
        0.00,
        1.01,
        0.05
    ):

        adjusted = group_rank_adjustment(
            base_score,
            groups,
            alpha
        )

        for rate in [
            0.36,
            0.37,
            0.38,
            0.39,
            0.40,
            0.41,
            0.42,
            0.43,
            0.44
        ]:

            n = round(
                len(T) * rate
            )

            order = np.argsort(
                -adjusted
            )

            pred = np.zeros(
                len(T),
                dtype=int
            )

            pred[
                order[:n]
            ] = 1

            candidates.append({
                "strategy": score_name,
                "alpha": alpha,
                "rate": rate,
                "score": adjusted,
                "prediction": pred
            })


# ============================================================
# 12. SAUVEGARDER TOUS LES CANDIDATS
# ============================================================

import os

os.makedirs(
    "candidates",
    exist_ok=True
)

summary = []

for i, candidate in enumerate(candidates):

    filename = (
        f"candidates/"
        f"{i:04d}_"
        f"{candidate['strategy']}_"
        f"a{candidate['alpha']:.2f}_"
        f"r{candidate['rate']:.2f}.csv"
    )

    submission = pd.DataFrame({
        "id_candidat":
            test["id_candidat"],
        "decision_octroi":
            candidate["prediction"]
    })

    submission.to_csv(
        filename,
        index=False
    )

    summary.append({
        "id": i,
        "strategy": candidate["strategy"],
        "alpha": candidate["alpha"],
        "rate": candidate["rate"],
        "n_awards":
            candidate["prediction"].sum()
    })


summary = pd.DataFrame(summary)

summary.to_csv(
    "candidates/INDEX.csv",
    index=False
)


# ============================================================
# 13. SOUMISSION PAR DÉFAUT
# ============================================================

# On commence avec :
# - score mérite_besoin
# - correction régionale modérée
# - 40 %

selected = None

for candidate in candidates:

    if (
        candidate["strategy"] == "merite_besoin"
        and abs(candidate["alpha"] - 0.25) < 1e-9
        and abs(candidate["rate"] - 0.40) < 1e-9
    ):

        selected = candidate
        break


final_submission = pd.DataFrame({
    "id_candidat":
        test["id_candidat"],
    "decision_octroi":
        selected["prediction"]
})

final_submission.to_csv(
    "predictions.csv",
    index=False
)


# ============================================================
# 14. RAPPORT
# ============================================================

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)

print(
    f"Nombre de candidats : {len(test)}"
)

print(
    f"Nombre d'octrois : "
    f"{selected['prediction'].sum()}"
)

print(
    f"Taux d'octroi : "
    f"{selected['prediction'].mean() * 100:.2f}%"
)

print(
    "\nFichier principal : predictions.csv"
)

print(
    "Candidats supplémentaires : candidates/"
)

print(
    "Index des candidats : candidates/INDEX.csv"
)