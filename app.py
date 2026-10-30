from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
EXT = RES / "external_validation"
PLOTS = RES / "plots"

st.set_page_config(page_title="Colorectal Adenoma Classifier", page_icon="🧬", layout="wide")

st.title("🧬 Colorectal Adenoma vs Normal Mucosa — Gene-Expression Classifier")
st.caption("Machine learning on GEO microarray data · training: GSE8671 · external validation: GSE50114 · research prototype, not a diagnostic tool")

page = st.sidebar.radio("Section", [
    "1. Summary",
    "2. Data & Method",
    "3. Internal Results",
    "4. External Validation",
    "5. Biology: Top Genes",
    "6. Explore Predictions",
    "7. Limitations",
])
st.sidebar.markdown("---")
st.sidebar.caption("Kunal Sah · VIT Vellore")


def load(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def show(path, caption=None, **kw):
    if path.exists():
        st.image(str(path), caption=caption, **kw)
    else:
        st.warning(f"Missing figure: {path.name}. Run scripts/external_validation_harmonized.py")


m = load(EXT / "metrics_harmonized.csv")

if page.startswith("1"):
    st.header("Summary")
    st.write(
        "Can gene expression separate colorectal **adenomas** (pre-cancerous polyps) from **normal colon mucosa**, "
        "and does that hold on data from a *different lab and microarray platform*?"
    )
    if m is not None:
        orig, harm = m.iloc[1], m.iloc[2]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Internal CV accuracy", f"{m.iloc[0].accuracy:.0%}", help="Patient-grouped 5-fold CV on GSE8671")
        c2.metric("External acc. (original)", f"{orig.accuracy:.0%}")
        c3.metric("External acc. (harmonized)", f"{harm.accuracy:.0%}", delta=f"{(harm.accuracy-orig.accuracy)*100:+.0f} pts")
        c4.metric("External ROC AUC", f"{harm.roc_auc:.2f}")
    st.markdown("""
**Key finding.** A model with a perfect internal score failed on the external cohort (24% accuracy) even though its
ROC AUC was 1.0. The ranking of samples was correct; the *scale* of the two platforms differed
(Affymetrix raw intensities vs Agilent log-ratios), so the 0.5 threshold was meaningless. Log-transforming and
standardizing each gene **within each dataset** fixed this without using any labels, raising external accuracy to ~93%
with 100% precision and 92% sensitivity.
""")
    st.info("Perfect internal scores on 64 samples are a warning sign, not a result. The external test is what makes this study credible.")

elif page.startswith("2"):
    st.header("Data & Method")
    a, b = st.columns(2)
    with a:
        st.subheader("Training — GSE8671")
        st.markdown("- Affymetrix HG-U133 Plus 2.0\n- 32 patients, paired normal + adenoma = 64 samples\n- Mapped to 10,784 genes shared with the external set")
    with b:
        st.subheader("External — GSE50114")
        st.markdown("- Agilent 4x44K (GPL6480)\n- 46 samples: 37 adenoma, 9 normal\n- Independent cohort, never seen in training")
    st.subheader("Pipeline")
    st.markdown("""
1. Map probes to gene symbols and keep genes common to both platforms.
2. **Harmonize**: log2 (Affymetrix only) then z-score each gene within its own dataset. No labels used.
3. Inside every CV fold: select top-50 genes (ANOVA F-test) → standardize → L2 logistic regression.
4. **Patient-grouped** cross-validation so a patient's normal and adenoma samples never straddle train/test.
5. Fit on all GSE8671, freeze, then predict GSE50114 once.
""")

elif page.startswith("3"):
    st.header("Internal Results (GSE8671, patient-grouped CV)")
    t = load(RES / "model_comparison.csv")
    if t is not None:
        st.dataframe(t.style.format({c: "{:.3f}" for c in t.columns if c != "Model"}), use_container_width=True)
    cols = st.columns(3)
    for col, (n, f) in zip(cols, [("Logistic Regression", "logistic_regression"), ("SVM", "svm"), ("Random Forest", "random_forest")]):
        with col:
            show(PLOTS / f"{f}_confusion_matrix.png", n)
    st.warning("All three models score 1.00. With 64 samples and a very strong tumor-vs-normal signal this is plausible, "
               "but it means internal CV cannot rank the models. Only external data can show whether the result generalizes.")

elif page.startswith("4"):
    st.header("External Validation (GSE50114)")
    if m is not None:
        t = m.copy()
        t["evaluation"] = ["Internal CV (GSE8671)", "External — original pipeline", "External — harmonized"]
        st.dataframe(t.set_index("evaluation").style.format("{:.3f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        show(EXT / "fig_before_after.png")
    with c2:
        show(EXT / "fig_probability_shift.png")
    c3, c4 = st.columns(2)
    with c3:
        show(EXT / "fig_roc.png", width=380)
    with c4:
        show(EXT / "fig_confusion_harmonized.png", width=380)
    st.markdown("""
**Reading the plots.** Both versions have AUC = 1.0: tumors score higher than normals in either case. Before harmonization
almost every probability sits near 0, so a 0.5 cutoff calls nearly everything *Normal* (sensitivity 5%). After harmonization
the two classes separate around 0.5.
""")
    st.caption("Caveat: the external z-scoring uses the external set's own gene means/SDs (label-free, but transductive) and that set is 80% adenoma. "
               "A deployable version would need reference-based batch correction.")

elif page.startswith("5"):
    st.header("Biology: Top Genes")
    g = load(EXT / "top_genes_harmonized.csv")
    c1, c2 = st.columns([1, 1])
    with c1:
        show(EXT / "fig_top_genes.png")
    with c2:
        if g is not None:
            st.dataframe(g.head(25), use_container_width=True, hide_index=True)
    st.markdown("""
Several selected genes are known Wnt-pathway components or targets (e.g. **AXIN2**, **RNF43**), consistent with the central
role of Wnt signalling in adenoma formation. This is a plausibility check, not proof that the classifier learned causal biology;
with 64 training samples, many of the 50 selected genes are interchangeable correlates.
""")

elif page.startswith("6"):
    st.header("Explore Predictions")
    which = st.radio("Dataset", ["External GSE50114 (harmonized)", "Internal out-of-fold (GSE8671)"], horizontal=True)
    if which.startswith("External"):
        df = load(EXT / "external_predictions_harmonized.csv")
    else:
        mod = st.selectbox("Model", ["logistic_regression", "svm", "random_forest"])
        df = load(RES / f"{mod}_predictions.csv")
    if df is not None:
        truth = "true_group" if "true_group" in df else "actual"
        pred = "predicted_group" if "predicted_group" in df else "predicted"
        df["correct"] = df[truth] == df[pred]
        only_wrong = st.checkbox("Show only misclassified samples")
        st.dataframe(df[~df["correct"]] if only_wrong else df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "predictions.csv", "text/csv")

elif page.startswith("7"):
    st.header("Limitations")
    st.markdown("""
- **Small data**: 64 training and 46 external samples, from two public studies.
- **Class imbalance externally**: 9 normal vs 37 adenoma; specificity (100%) rests on only 9 samples.
- **Perfect internal scores** cannot rank the three models; they mainly show the task is easy within one platform.
- **Harmonization is transductive**: it needs the new dataset's gene statistics. It does not yet work for a single new patient.
- **Task scope**: adenoma vs normal mucosa, not cancer staging or screening.
- **Not clinical**: no prospective validation; research prototype only.

**Future work**: reference-based batch correction (ComBat / quantile normalization to a fixed reference), a third independent cohort,
adenoma-vs-carcinoma classification, and pathway enrichment of selected genes.
""")

st.divider()
st.caption("Research prototype · GEO GSE8671 & GSE50114 · Not validated for clinical use")
