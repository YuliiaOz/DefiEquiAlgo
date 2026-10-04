import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

TRAIN = "data/donnees_demandes.csv"
TEST = "data/candidats_evaluation.csv"

AWARD_RATE = 0.40
RANDOM_STATE = 42

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

y = train["decision_octroi"].astype(int)

X = train.drop(columns=["decision_octroi"]).copy()
T = test.copy()


# ============================================================
# 1. GROUPES D'ÉQUITÉ
# ============================================================

REMOTE = {
    "Bas-Saint-Laurent",
    "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine"
}

for data in [X, T]:

    data["groupe"] = np.where(
        data["region_administrative"].isin(REMOTE),
        "eloignee",
        "centre"
    )


# ============================================================
# 2. SCORE DE MÉRITE
# ============================================================

def percentile(x):
    return pd.Series(x).rank(pct=True).values


def build_merit_score(data):

    cote = percentile(
        data["cote_r_equivalent"]
    )

    # Plus faible revenu = plus grand besoin
    revenu = 1 - percentile(
        data["revenu_familial_estime"]
    )

    # Plus d'heures de travail = contrainte plus importante
    heures = percentile(
        data["heures_travail_semaine"]
    )

    # Plus grande distance = contrainte géographique plus importante
    distance = percentile(
        data["distance_domicile_campus_km"]
    )

    premiere_gen = (
        data["premiere_generation_universitaire"]
        .astype(float)
    )

    return (
        0.55 * cote
        + 0.15 * revenu
        + 0.12 * heures
        + 0.10 * distance
        + 0.08 * premiere_gen
    )


X["score_merite"] = build_merit_score(X)
T["score_merite"] = build_merit_score(T)


# ============================================================
# 3. SCORE ACADÉMIQUE
# ============================================================

X["score_academique"] = percentile(
    X["cote_r_equivalent"]
)

T["score_academique"] = percentile(
    T["cote_r_equivalent"]
)


# ============================================================
# 4. SCORE DE BESOIN
# ============================================================

def build_need_score(data):

    revenu = 1 - percentile(
        data["revenu_familial_estime"]
    )

    heures = percentile(
        data["heures_travail_semaine"]
    )

    distance = percentile(
        data["distance_domicile_campus_km"]
    )

    premiere_gen = (
        data["premiere_generation_universitaire"]
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
# 5. VARIABLES UTILISÉES PAR LE MODÈLE
# ============================================================

features = [
    "cote_r_equivalent",
    "revenu_familial_estime",
    "heures_travail_semaine",
    "distance_domicile_campus_km",
    "premiere_generation_universitaire",
    "score_merite",
    "score_academique",
    "score_besoin"
]

X_model = X[features].copy()
T_model = T[features].copy()


# ============================================================
# 6. MODÈLE
# ============================================================

model = ExtraTreesClassifier(
    n_estimators=2500,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features=0.9,
    criterion="entropy",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# ============================================================
# 7. VALIDATION CROISÉE OOF
# ============================================================

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof = np.zeros(len(X))

for fold, (tr, va) in enumerate(
    skf.split(X_model, y),
    start=1
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
# 8. PERFORMANCE HISTORIQUE
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE DU MODÈLE")
print("=" * 70)

oof_pred = (
    oof >= 0.50
).astype(int)

print(
    f"Accuracy : "
    f"{accuracy_score(y, oof_pred):.4f}"
)

print(
    f"F1-score : "
    f"{f1_score(y, oof_pred):.4f}"
)


# ============================================================
# 9. APPRENTISSAGE FINAL
# ============================================================

model.fit(
    X_model,
    y
)

test_probability = model.predict_proba(
    T_model
)[:, 1]


# ============================================================
# 10. SCORES LATENTS
# ============================================================

merit = T["score_merite"].to_numpy()
academic = T["score_academique"].to_numpy()
need = T["score_besoin"].to_numpy()


# ============================================================
# 11. SCORE FINAL DE BASE
# ============================================================

# Le modèle reste le signal principal.
# Les scores explicites ajoutent une dimension
# de mérite et de besoin socio-économique.

score_base = (
    0.50 * test_probability
    + 0.30 * merit
    + 0.10 * academic
    + 0.10 * need
)


# ============================================================
# 12. CORRECTION PAR RANG INTRA-GROUPE
# ============================================================

def group_rank_adjustment(
    score,
    groups,
    alpha
):

    score = np.asarray(score)

    # Classement global
    global_rank = (
        pd.Series(score)
        .rank(pct=True)
        .to_numpy()
    )

    # Classement à l'intérieur de chaque groupe
    group_rank = np.zeros(len(score))

    for group in np.unique(groups):

        idx = np.where(
            groups == group
        )[0]

        group_rank[idx] = (
            pd.Series(score[idx])
            .rank(pct=True)
            .to_numpy()
        )

    # alpha = 0 :
    # classement totalement global
    #
    # alpha = 1 :
    # classement totalement intra-groupe

    return (
        (1 - alpha) * global_rank
        + alpha * group_rank
    )


groups = T["groupe"].to_numpy()


# ============================================================
# 13. FRONT DE PARETO SUR LES DONNÉES HISTORIQUES
# ============================================================
#
# IMPORTANT :
# L'accuracy doit être calculée sur les données historiques,
# car ce sont les seules données pour lesquelles nous avons
# une décision historique y.
#
# Le jeu TEST (4 000 candidats) sert uniquement à produire
# la soumission finale.
# ============================================================

pareto_results = []

alphas = np.arange(
    0.00,
    1.01,
    0.05
)

# ------------------------------------------------------------
# Score historique de base
# ------------------------------------------------------------

score_historique = (
    0.50 * oof
    + 0.30 * X["score_merite"].to_numpy()
    + 0.10 * X["score_academique"].to_numpy()
    + 0.10 * X["score_besoin"].to_numpy()
)

groups_historique = X["groupe"].to_numpy()

# Budget historique : 40 %
n_awards_historique = round(
    len(X) * AWARD_RATE
)

for alpha in alphas:

    adjusted_score = group_rank_adjustment(
        score_historique,
        groups_historique,
        alpha
    )

    # --------------------------------------------------------
    # Sélection des 40 % meilleurs
    # --------------------------------------------------------

    order = np.argsort(
        -adjusted_score
    )

    prediction = np.zeros(
        len(X),
        dtype=int
    )

    prediction[
        order[:n_awards_historique]
    ] = 1

    # --------------------------------------------------------
    # Performance historique
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        prediction
    )

    f1 = f1_score(
        y,
        prediction
    )

    # --------------------------------------------------------
    # Parité démographique
    # --------------------------------------------------------

    centre_rate = prediction[
        groups_historique == "centre"
    ].mean()

    remote_rate = prediction[
        groups_historique == "eloignee"
    ].mean()

    parity_gap = abs(
        centre_rate - remote_rate
    )

    pareto_results.append({

        "alpha": alpha,

        "taux_octroi":
            prediction.mean(),

        "accuracy":
            accuracy,

        "f1":
            f1,

        "taux_centre":
            centre_rate,

        "taux_eloignee":
            remote_rate,

        "parity_gap":
            parity_gap
    })


pareto = pd.DataFrame(
    pareto_results
)


# ============================================================
# 14. AFFICHAGE DU FRONT DE PARETO
# ============================================================

print("\n" + "=" * 70)
print("FRONT DE PARETO")
print("=" * 70)

print(
    pareto[
        [
            "alpha",
            "taux_octroi",
            "accuracy",
            "f1",
            "taux_centre",
            "taux_eloignee",
            "parity_gap"
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 15. GRAPHIQUE DU FRONT DE PARETO
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    pareto["parity_gap"],
    pareto["accuracy"],
    marker="o"
)

plt.xlabel(
    "Écart de parité démographique"
)

plt.ylabel(
    "Accuracy historique"
)

plt.title(
    "Front de Pareto — Équité vs utilité"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "pareto_front.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. CHOIX DU MODÈLE FINAL
# ============================================================

# Accuracy avec alpha = 0
accuracy_reference = pareto.loc[
    pareto["alpha"] == 0,
    "accuracy"
].iloc[0]

# On accepte une perte maximale de 3 points
max_accuracy_loss = 0.03

acceptable = pareto[
    pareto["accuracy"]
    >=
    accuracy_reference - max_accuracy_loss
].copy()

# Parmi les solutions acceptables,
# on choisit celle avec le plus petit écart de parité.

selected_row = acceptable.loc[
    acceptable["parity_gap"].idxmin()
]

selected_alpha = float(
    selected_row["alpha"]
)

print(
    f"\nAlpha sélectionné : "
    f"{selected_alpha:.2f}"
)


# ============================================================
# 17. APPLICATION AU JEU D'ÉVALUATION
# ============================================================

# ------------------------------------------------------------
# Maintenant seulement, on applique la stratégie choisie
# aux 4 000 candidats du jeu TEST.
# ------------------------------------------------------------

final_score = group_rank_adjustment(
    score_base,
    groups,
    selected_alpha
)

order = np.argsort(
    -final_score
)

final_prediction = np.zeros(
    len(T),
    dtype=int
)

n_awards = round(
    len(T) * AWARD_RATE
)

final_prediction[
    order[:n_awards]
] = 1

# ============================================================
# 18. CRÉATION DE predictions.csv
# ============================================================

final_submission = pd.DataFrame({
    "id_candidat": test["id_candidat"],
    "decision_octroi": final_prediction
})

final_submission.to_csv(
    "predictions.csv",
    index=False
)


# ============================================================
# 19. VALIDATION DE LA SOUMISSION
# ============================================================

assert len(final_submission) == 4000

assert list(final_submission.columns) == [
    "id_candidat",
    "decision_octroi"
]

assert set(
    final_submission["decision_octroi"].unique()
).issubset({0, 1})

award_rate = (
    final_submission["decision_octroi"].mean()
)

assert 0.36 <= award_rate <= 0.44


# ============================================================
# 20. RAPPORT FINAL
# ============================================================

centre_final = final_prediction[
    groups == "centre"
].mean()

remote_final = final_prediction[
    groups == "eloignee"
].mean()

final_gap = abs(
    centre_final - remote_final
)


print("\n" + "=" * 70)
print("SOUMISSION FINALE")
print("=" * 70)

print(
    f"Alpha sélectionné : {selected_alpha:.2f}"
)

print(
    f"Nombre de candidats : {len(test)}"
)

print(
    f"Nombre d'octrois : {final_prediction.sum()}"
)

print(
    f"Taux d'octroi : {award_rate * 100:.2f}%"
)

print(
    f"Taux Centre : {centre_final * 100:.2f}%"
)

print(
    f"Taux Éloignée : {remote_final * 100:.2f}%"
)

print(
    f"Écart de parité : {final_gap:.4f}"
)

print(
    "\nFichiers générés :"
)

print(
    "  ✓ predictions.csv"
)

print(
    "  ✓ pareto_front.png"
)