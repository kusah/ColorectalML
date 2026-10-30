"""
Cross-platform external validation: GSE8671 (Affymetrix, raw intensity) -> GSE50114 (Agilent, log-ratio).

Finding: a model trained on raw GSE8671 scores perfectly inside the dataset but
fails on GSE50114 (accuracy 23.9%, sensitivity 5.4%) even though ROC AUC = 1.0.
The ranking is right; the decision threshold is wrong because the two platforms
use different measurement scales (platform batch effect).

Fix: log2-transform the Affymetrix data and z-score every gene WITHIN each dataset.
No class labels are used for this step. The classifier is trained only on GSE8671.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
GL = ROOT / "data" / "processed" / "gene_level"
OUT = ROOT / "results" / "external_validation"
OUT.mkdir(parents=True, exist_ok=True)
SEED, TOP_K, C = 42, 50, 0.1


def zscore(X):
    return ((X - X.mean()) / X.std().replace(0, 1)).fillna(0.0)


def metrics(y, pred, prob):
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred),
        "specificity": recall_score(y, pred, pos_label=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_classif, k=TOP_K)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=SEED)),
    ])


# ---- load ----
Xt = pd.read_csv(GL / "X_train.csv", index_col="sample_id")
Xe = pd.read_csv(GL / "X_external.csv", index_col="sample_id")
Xt.index, Xe.index = Xt.index.astype(str), Xe.index.astype(str)
yt = pd.read_csv(GL / "y_train.csv", index_col="sample_id").loc[Xt.index, "label"].astype(int)
meta_e = pd.read_csv(ROOT / "data/processed/GSE50114/sample_metadata.csv",
                     dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id").loc[Xe.index]
ye = meta_e["tissue"].map({"Normal": 0, "Adenoma": 1}).astype(int)
lab = pd.read_csv(ROOT / "data/processed/sample_labels.csv", dtype={"sample_id": str}
                  ).drop_duplicates("sample_id").set_index("sample_id")
groups = lab.loc[Xt.index, "patient_id"]

# ---- harmonize ----
Xt_h = zscore(np.log2(Xt.clip(lower=1)))
Xe_h = zscore(Xe)

rows = []
# (A) internal grouped CV on harmonized training data
cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
oof = cross_val_predict(make_pipe(), Xt_h, yt, groups=groups, cv=cv, method="predict_proba")[:, 1]
rows.append({"evaluation": "Internal: patient-grouped 5-fold CV (GSE8671)", **metrics(yt, (oof > .5).astype(int), oof)})

# (B) baseline = the ORIGINAL pipeline's saved external predictions (scripts/train_gene_level_model.py)
orig = pd.read_csv(OUT / "external_predictions.csv").set_index("sample_id").loc[Xe.index]
pn = orig["adenoma_probability"].to_numpy()
rows.append({"evaluation": "External GSE50114 - original pipeline (no harmonization)",
             **metrics(ye, orig["predicted_label"].to_numpy(), pn)})

# (C) harmonized external
final = make_pipe().fit(Xt_h, yt)
ph = final.predict_proba(Xe_h)[:, 1]
pred_h = (ph > .5).astype(int)
rows.append({"evaluation": "External GSE50114 - harmonized (log2 + per-dataset z-score)", **metrics(ye, pred_h, ph)})

res = pd.DataFrame(rows)
res.to_csv(OUT / "metrics_harmonized.csv", index=False)
print(res.round(3).to_string(index=False))

pd.DataFrame({"sample_id": Xe.index, "true_group": ye.map({0: "Normal", 1: "Adenoma"}).values,
              "predicted_group": np.where(pred_h == 1, "Adenoma", "Normal"),
              "adenoma_probability": ph}).to_csv(OUT / "external_predictions_harmonized.csv", index=False)

# top genes
sel = final.named_steps["select"].get_support()
coefs = pd.DataFrame({"gene": Xt.columns[sel], "coefficient": final.named_steps["model"].coef_[0]})
coefs["direction"] = np.where(coefs.coefficient > 0, "up in adenoma", "down in adenoma")
coefs.reindex(coefs.coefficient.abs().sort_values(ascending=False).index).to_csv(
    OUT / "top_genes_harmonized.csv", index=False)

import joblib
joblib.dump(final, ROOT / "models" / "gene_level_harmonized_model.joblib")

# ---- figures ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 1. before/after
fig, ax = plt.subplots(figsize=(7, 4.2))
names = ["Accuracy", "Balanced acc.", "Sensitivity"]
keys = ["accuracy", "balanced_accuracy", "sensitivity"]
x = np.arange(3); w = .36
b1 = ax.bar(x - w/2, [rows[1][k] for k in keys], w, label="Original pipeline", color="#c0504d")
b2 = ax.bar(x + w/2, [rows[2][k] for k in keys], w, label="Harmonized", color="#2e7d5b")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + .015, f"{r.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.22); ax.set_ylabel("Score")
ax.set_title("External validation on GSE50114: effect of platform harmonization")
ax.legend(frameon=False, ncol=2, loc="upper center"); fig.tight_layout(); fig.savefig(OUT / "fig_before_after.png", dpi=200); plt.close(fig)

# 2. ROC
fig, ax = plt.subplots(figsize=(4.8, 4.5))
for p, lbl, c in [(pn, "Naive", "#c0504d"), (ph, "Harmonized", "#2e7d5b")]:
    fpr, tpr, _ = roc_curve(ye, p); ax.plot(fpr, tpr, label=f"{lbl} (AUC {roc_auc_score(ye, p):.2f})", color=c, lw=2)
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="External ROC (GSE50114)")
ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig_roc.png", dpi=200); plt.close(fig)

# 3. probability distribution
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, p, t in [(axs[0], pn, "Naive"), (axs[1], ph, "Harmonized")]:
    ax.hist(p[ye.values == 0], bins=np.linspace(0, 1, 21), alpha=.7, label="Normal", color="#3b6ea5")
    ax.hist(p[ye.values == 1], bins=np.linspace(0, 1, 21), alpha=.7, label="Adenoma", color="#d98c2b")
    ax.axvline(.5, color="k", ls="--", lw=1); ax.set_title(f"{t}: predicted P(adenoma)"); ax.set_xlabel("Probability")
axs[0].set_ylabel("Samples"); axs[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_probability_shift.png", dpi=200); plt.close(fig)

# 4. confusion matrix
cm = confusion_matrix(ye, pred_h)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Adenoma"], yticklabels=["Normal", "Adenoma"],
       xlabel="Predicted", ylabel="True", title="GSE50114 (harmonized)")
fig.tight_layout(); fig.savefig(OUT / "fig_confusion_harmonized.png", dpi=200); plt.close(fig)

# 5. top genes
top = coefs.reindex(coefs.coefficient.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(top.gene, top.coefficient, color=np.where(top.coefficient > 0, "#d98c2b", "#3b6ea5"))
ax.set_xlabel("Logistic regression coefficient"); ax.set_title("Top 15 genes (orange = higher in adenoma)")
fig.tight_layout(); fig.savefig(OUT / "fig_top_genes.png", dpi=200); plt.close(fig)

json.dump({"top_k": TOP_K, "C": C, "n_train": int(len(yt)), "n_external": int(len(ye)),
           "external_class_counts": {"Normal": int((ye == 0).sum()), "Adenoma": int((ye == 1).sum())}},
          open(OUT / "run_config.json", "w"), indent=2)
print("done")
