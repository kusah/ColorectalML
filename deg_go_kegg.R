# =============================================================================
# GSE8671 differential expression + GO / KEGG enrichment (R workflow)
# -----------------------------------------------------------------------------
# Run from the project root:   Rscript scripts/deg_go_kegg.R
#
# Computes DEGs directly with limma (no GEO2R export needed), then produces:
#   volcano plot, top-50 DEG heatmap, GO BP enrichment, KEGG enrichment,
#   and the full result tables. All outputs go to results/deg/.
#
# Group and patient labels come from data/processed/sample_labels.csv
# (32 patients, each with one normal mucosa and one adenoma sample), so the
# default model is a paired design (~ patient + group).
# =============================================================================

# ---- 0. Packages (installed only if missing) --------------------------------
cran_pkgs <- c("BiocManager", "ggplot2", "pheatmap")
bioc_pkgs <- c("GEOquery", "limma", "clusterProfiler", "org.Hs.eg.db", "enrichplot")

for (p in cran_pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
for (p in bioc_pkgs) if (!requireNamespace(p, quietly = TRUE)) BiocManager::install(p, update = FALSE, ask = FALSE)

suppressPackageStartupMessages({
  library(GEOquery); library(limma); library(ggplot2); library(pheatmap)
  library(clusterProfiler); library(org.Hs.eg.db); library(enrichplot)
})

# ---- 1. Settings -------------------------------------------------------------
GSE_ID      <- "GSE8671"
ADJP_CUTOFF <- 0.05
LOGFC_CUT   <- 1
TOP_N       <- 50
PAIRED      <- TRUE            # TRUE: ~ patient + group;  FALSE: ~ group (like GEO2R default)
LABELS_CSV  <- "data/processed/sample_labels.csv"
OUT_DIR     <- "results/deg"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- 2. Download expression matrix and probe annotation ----------------------
gset <- getGEO(GSE_ID, GSEMatrix = TRUE, getGPL = TRUE)
if (length(gset) > 1) gset <- gset[[grep("GPL570", names(gset))[1]]] else gset <- gset[[1]]
expr <- exprs(gset)
cat("Expression matrix:", nrow(expr), "probes x", ncol(expr), "samples\n")

# GEO series matrices are usually log2-scale already; log2 only if values look raw
qx <- as.numeric(quantile(expr, c(0, 0.25, 0.5, 0.75, 0.99, 1), na.rm = TRUE))
if (qx[5] > 100 || (qx[6] - qx[1] > 50 && qx[2] > 0)) {
  expr[expr <= 0] <- NA
  expr <- log2(expr)
  cat("Applied log2 transform.\n")
}

# ---- 3. Sample labels --------------------------------------------------------
labels <- read.csv(LABELS_CSV, stringsAsFactors = FALSE)
labels <- labels[match(colnames(expr), labels$sample_id), ]
stopifnot(!anyNA(labels$sample_id))
group <- factor(ifelse(labels$group == "Normal", "Normal", "Adenoma"),
                levels = c("Normal", "Adenoma"))
patient <- factor(labels$patient_id)
cat("Groups:\n"); print(table(group))

# ---- 4. limma differential expression (Adenoma vs Normal) --------------------
design <- if (PAIRED) model.matrix(~ patient + group) else model.matrix(~ group)
fit <- eBayes(lmFit(expr, design))
coef_name <- "groupAdenoma"
tt <- topTable(fit, coef = coef_name, number = Inf, adjust.method = "BH", sort.by = "P")
tt$ID <- rownames(tt)

# Probe -> gene symbol from the platform annotation
fdat <- fData(gset)
sym_col <- grep("^gene.?symbol$", colnames(fdat), ignore.case = TRUE, value = TRUE)[1]
tt$Gene.symbol <- fdat[tt$ID, sym_col]
# Multiple symbols per probe are separated by "///": keep the first
tt$Gene.symbol <- trimws(sub("///.*$", "", tt$Gene.symbol))

deg <- tt
write.csv(deg, file.path(OUT_DIR, "GSE8671_limma_all_probes.csv"), row.names = FALSE)

# ---- 5. DEG filtering --------------------------------------------------------
sig_deg <- subset(deg, adj.P.Val < ADJP_CUTOFF & abs(logFC) >= LOGFC_CUT)
cat("Significant probes (adj.P <", ADJP_CUTOFF, ", |logFC| >=", LOGFC_CUT, "):", nrow(sig_deg), "\n")

sig_deg <- sig_deg[!is.na(sig_deg$Gene.symbol) & sig_deg$Gene.symbol != "", ]
sig_deg <- sig_deg[order(sig_deg$adj.P.Val), ]
# one row (best p-value) per gene so counts are gene-level
sig_deg <- sig_deg[!duplicated(sig_deg$Gene.symbol), ]
cat("Significant DEGs (unique genes with symbol):", nrow(sig_deg),
    "| up:", sum(sig_deg$logFC > 0), "| down:", sum(sig_deg$logFC < 0), "\n")
write.csv(sig_deg, file.path(OUT_DIR, "GSE8671_significant_DEGs.csv"), row.names = FALSE)

top50 <- head(sig_deg, TOP_N)
write.csv(top50, file.path(OUT_DIR, "GSE8671_top50_DEGs.csv"), row.names = FALSE)

# ---- 6. Volcano plot ---------------------------------------------------------
deg$Status <- "Not Significant"
deg$Status[deg$adj.P.Val < ADJP_CUTOFF & deg$logFC >=  LOGFC_CUT] <- "Upregulated"
deg$Status[deg$adj.P.Val < ADJP_CUTOFF & deg$logFC <= -LOGFC_CUT] <- "Downregulated"
deg$negLogAdjP <- -log10(deg$adj.P.Val)

p_volcano <- ggplot(deg, aes(x = logFC, y = negLogAdjP, color = Status)) +
  geom_point(alpha = 0.7, size = 1.5) +
  scale_color_manual(values = c("Upregulated" = "red", "Downregulated" = "blue",
                                "Not Significant" = "grey")) +
  geom_vline(xintercept = c(-LOGFC_CUT, LOGFC_CUT), linetype = "dashed") +
  geom_hline(yintercept = -log10(ADJP_CUTOFF), linetype = "dashed") +
  labs(title = "Volcano Plot of GSE8671 (Adenoma vs Normal Mucosa)",
       x = "log2 Fold Change", y = "-log10 Adjusted P-value") +
  theme_minimal()
ggsave(file.path(OUT_DIR, "volcano_plot.png"), p_volcano, width = 8, height = 6, dpi = 300)

# ---- 7. Top-50 DEG heatmap ---------------------------------------------------
expr_top <- expr[top50$ID, , drop = FALSE]
rownames(expr_top) <- make.unique(top50$Gene.symbol)
annotation_col <- data.frame(Group = ifelse(group == "Normal", "Normal Mucosa", "Tumor / Adenoma"))
rownames(annotation_col) <- colnames(expr_top)

png(file.path(OUT_DIR, "heatmap_top50_DEGs.png"), width = 2400, height = 2200, res = 250)
pheatmap(expr_top, scale = "row", annotation_col = annotation_col,
         show_rownames = TRUE, show_colnames = FALSE, fontsize_row = 8,
         clustering_distance_rows = "euclidean", clustering_distance_cols = "euclidean",
         clustering_method = "complete",
         main = "Heatmap of Top 50 Differentially Expressed Genes in GSE8671")
dev.off()

# ---- 8. Gene symbol -> Entrez ID --------------------------------------------
gene_map <- bitr(unique(sig_deg$Gene.symbol), fromType = "SYMBOL",
                 toType = "ENTREZID", OrgDb = org.Hs.eg.db)
entrez_ids <- unique(gene_map$ENTREZID)
cat("Mapped Entrez IDs:", length(entrez_ids), "of", length(unique(sig_deg$Gene.symbol)), "symbols\n")
write.csv(gene_map, file.path(OUT_DIR, "symbol_to_entrez_map.csv"), row.names = FALSE)

# Background = all annotated genes tested, which is the correct universe for a microarray
universe_sym <- unique(deg$Gene.symbol[!is.na(deg$Gene.symbol) & deg$Gene.symbol != ""])
universe_map <- bitr(universe_sym, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
universe_ids <- unique(universe_map$ENTREZID)

# ---- 9. GO Biological Process enrichment -------------------------------------
ego <- enrichGO(gene = entrez_ids, universe = universe_ids, OrgDb = org.Hs.eg.db,
                ont = "BP", pAdjustMethod = "BH", pvalueCutoff = 0.05,
                qvalueCutoff = 0.05, readable = TRUE)
ego_df <- as.data.frame(ego)
write.csv(ego_df, file.path(OUT_DIR, "GO_BP_enrichment.csv"), row.names = FALSE)
if (nrow(ego_df) > 0) {
  p_go <- dotplot(ego, showCategory = 15) + ggtitle("GO Biological Process Enrichment - GSE8671")
  ggsave(file.path(OUT_DIR, "GO_BP_dotplot.png"), p_go, width = 9, height = 7, dpi = 300)
} else cat("No significant GO terms.\n")

# ---- 10. KEGG pathway enrichment (needs internet access to KEGG) -------------
ekegg <- tryCatch(
  enrichKEGG(gene = entrez_ids, universe = universe_ids, organism = "hsa", pvalueCutoff = 0.05),
  error = function(e) { message("KEGG query failed: ", conditionMessage(e)); NULL })

if (!is.null(ekegg) && nrow(as.data.frame(ekegg)) > 0) {
  ekegg <- setReadable(ekegg, OrgDb = org.Hs.eg.db, keyType = "ENTREZID")
  write.csv(as.data.frame(ekegg), file.path(OUT_DIR, "KEGG_enrichment.csv"), row.names = FALSE)
  p_kegg <- dotplot(ekegg, showCategory = 12) + ggtitle("KEGG Pathway Enrichment - GSE8671")
  ggsave(file.path(OUT_DIR, "KEGG_dotplot.png"), p_kegg, width = 9, height = 6, dpi = 300)
} else cat("No significant KEGG pathways (or KEGG unavailable).\n")

writeLines(capture.output(sessionInfo()), file.path(OUT_DIR, "sessionInfo.txt"))
cat("Done. Outputs in", OUT_DIR, "\n")
