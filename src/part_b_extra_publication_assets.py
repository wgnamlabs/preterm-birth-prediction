"""
Part B - Extra Publication Assets (Table 1, Forest plot, SHAP, DCA, Calibration)

Extracted from cdc-final-16-tri-u-ca.ipynb for readability / code review.
Run order: part_a -> part_b -> part_c (each reuses variables from the previous
one, so they must be executed in the same Python session / notebook kernel).
"""

# ============================================================
# PART B — EXTRA PUBLICATION FIGURES & TABLES (Q1-ready)
# Run this AFTER the main pipeline cell (reuses its variables)
# ============================================================

import itertools
from scipy.stats import chi2

EXTRA_DIR = os.path.join(OUT_DIR, "extra_publication_assets")
Path(EXTRA_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("PART B: EXTRA FIGURES/TABLES FOR MANUSCRIPT")
print("=" * 90)

# ------------------------------------------------------------
# B0. Helper: numeric bootstrap CI (returns floats, not string)
# ------------------------------------------------------------
def bootstrap_auc_ci_values(y_true, y_score, n_boot=200, max_n=200_000, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) > max_n:
        idx = rng.choice(len(y_true), max_n, replace=False)
        y_true, y_score = y_true[idx], y_score[idx]
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------
# B1. Table 1 — Baseline characteristics by outcome (with SMD)
# ------------------------------------------------------------
print("\n[B1] Building Table 1 (baseline characteristics)...")

def smd(x1, x0):
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    s1, s0 = np.nanstd(x1, ddof=1), np.nanstd(x0, ddof=1)
    pooled_sd = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
    return 0.0 if pooled_sd == 0 else float((m1 - m0) / pooled_sd)

X_full_for_table1 = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
y_full_for_table1 = pd.concat([y_train, y_val, y_test], axis=0).reset_index(drop=True)

table1_rows = []
for col in X_full_for_table1.columns:
    grp1 = X_full_for_table1.loc[y_full_for_table1 == 1, col].values
    grp0 = X_full_for_table1.loc[y_full_for_table1 == 0, col].values
    table1_rows.append({
        "Feature": col,
        "Preterm_Mean": float(np.nanmean(grp1)),
        "Preterm_SD": float(np.nanstd(grp1, ddof=1)),
        "Term_Mean": float(np.nanmean(grp0)),
        "Term_SD": float(np.nanstd(grp0, ddof=1)),
        "SMD": smd(grp1, grp0),
    })

table1_df = pd.DataFrame(table1_rows).sort_values("SMD", key=lambda s: s.abs(), ascending=False)
table1_path = os.path.join(EXTRA_DIR, "table1_baseline_characteristics.csv")
table1_df.to_csv(table1_path, index=False)
print(table1_df.head(15))
print("Saved:", table1_path)

del X_full_for_table1, y_full_for_table1
free_memory()

# ------------------------------------------------------------
# B2. Forest plot — AUROC with 95% CI, all full-test models
# ------------------------------------------------------------
print("\n[B2] Forest plot of AUROC (95% CI)...")

forest_rows = []
for name, pred in plot_preds_full.items():
    auc = roc_auc_score(y_test, pred)
    lo, hi = bootstrap_auc_ci_values(y_test, pred)
    forest_rows.append({"Model": name, "AUROC": auc, "CI_Low": lo, "CI_High": hi})

forest_df = pd.DataFrame(forest_rows).sort_values("AUROC")
forest_path = os.path.join(EXTRA_DIR, "auroc_forest_table.csv")
forest_df.to_csv(forest_path, index=False)

plt.figure(figsize=(8, 0.6 * len(forest_df) + 2), dpi=200)
y_pos = np.arange(len(forest_df))
plt.errorbar(
    forest_df["AUROC"], y_pos,
    xerr=[forest_df["AUROC"] - forest_df["CI_Low"], forest_df["CI_High"] - forest_df["AUROC"]],
    fmt="o", capsize=4, color="black", ecolor="gray",
)
plt.yticks(y_pos, forest_df["Model"])
plt.xlabel("AUROC (95% Bootstrap CI)")
plt.title("Forest Plot of AUROC Across Models — Full Test Set")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "auroc_forest_plot.png"), dpi=300)
plt.close()
print("Saved forest plot + table.")

# ------------------------------------------------------------
# B3. Confusion matrix heatmaps (grid, full-test models)
# ------------------------------------------------------------
print("\n[B3] Confusion matrix heatmaps...")

n_models = len(plot_preds_full)
n_cols = min(3, n_models)
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows), dpi=150)
axes = np.array(axes).reshape(-1)

for i, (name, pred) in enumerate(plot_preds_full.items()):
    thr = optimal_threshold(y_test, pred)
    y_pred = (np.asarray(pred) >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[i]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\n(threshold={thr:.3f})", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Term", "Preterm"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Term", "Preterm"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.suptitle("Confusion Matrices — Full Test Set (Youden-optimal threshold)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "confusion_matrices_grid.png"), dpi=300, bbox_inches="tight")
plt.close()
print("Saved confusion matrix grid.")

# ------------------------------------------------------------
# B4. Youden's J / operating-point metrics table
# ------------------------------------------------------------
print("\n[B4] Youden's J operating-point table...")

youden_rows = []
for name, pred in plot_preds_full.items():
    pred = np.asarray(pred)
    fpr, tpr, thr = roc_curve(y_test, pred)
    j = tpr - fpr
    best_idx = np.argmax(j)
    y_pred = (pred >= thr[best_idx]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    youden_rows.append({
        "Model": name,
        "Optimal_Threshold": float(thr[best_idx]),
        "Youden_J": float(j[best_idx]),
        "Sensitivity": float(tp / (tp + fn + 1e-8)),
        "Specificity": float(tn / (tn + fp + 1e-8)),
        "PPV": float(tp / (tp + fp + 1e-8)),
        "NPV": float(tn / (tn + fn + 1e-8)),
    })

youden_df = pd.DataFrame(youden_rows).sort_values("Youden_J", ascending=False)
youden_path = os.path.join(EXTRA_DIR, "youden_operating_point_table.csv")
youden_df.to_csv(youden_path, index=False)
print(youden_df)
print("Saved:", youden_path)

# ------------------------------------------------------------
# B5. Sensitivity/Specificity vs threshold — best model
# ------------------------------------------------------------
print("\n[B5] Sensitivity/Specificity vs threshold curve...")

best_model_name = forest_df.sort_values("AUROC", ascending=False)["Model"].iloc[0]
best_pred = np.asarray(plot_preds_full[best_model_name])

thr_grid = np.linspace(0.01, 0.99, 99)
sens_list, spec_list = [], []
for t in thr_grid:
    y_pred = (best_pred >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens_list.append(tp / (tp + fn + 1e-8))
    spec_list.append(tn / (tn + fp + 1e-8))

plt.figure(figsize=(8, 6), dpi=200)
plt.plot(thr_grid, sens_list, label="Sensitivity")
plt.plot(thr_grid, spec_list, label="Specificity")
plt.xlabel("Decision Threshold")
plt.ylabel("Metric Value")
plt.title(f"Sensitivity/Specificity vs Threshold — {best_model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "sens_spec_vs_threshold_best_model.png"), dpi=300)
plt.close()
print("Saved. Best model used:", best_model_name)

# ------------------------------------------------------------
# B6. Predicted probability distribution by class — best model
# ------------------------------------------------------------
print("\n[B6] Predicted probability distribution...")

plt.figure(figsize=(8, 6), dpi=200)
plt.hist(best_pred[np.asarray(y_test) == 0], bins=50, alpha=0.6, label="Term", density=True)
plt.hist(best_pred[np.asarray(y_test) == 1], bins=50, alpha=0.6, label="Preterm", density=True)
plt.xlabel("Predicted Probability of Preterm Birth")
plt.ylabel("Density")
plt.title(f"Predicted Probability Distribution — {best_model_name}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "predicted_probability_distribution.png"), dpi=300)
plt.close()
print("Saved.")

# ------------------------------------------------------------
# B7. SHAP dependence plots (top 6 features) + bar plot
# ------------------------------------------------------------
if RUN_SHAP and HAS_SHAP:
    print("\n[B7] SHAP dependence plots for top features...")
    try:
        shap_sample = X_test.sample(min(SHAP_SAMPLE_N, len(X_test)), random_state=SEED)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(shap_sample)

        # Mean |SHAP| bar plot
        plt.figure(figsize=(8, 8), dpi=200)
        shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False, max_display=20)
        plt.title("Mean |SHAP value| — Top 20 Features")
        plt.tight_layout()
        plt.savefig(os.path.join(EXTRA_DIR, "shap_bar_top20.png"), dpi=300, bbox_inches="tight")
        plt.close()

        top6_features = fi.head(6)["feature"].tolist()
        for feat in top6_features:
            if feat not in shap_sample.columns:
                continue
            plt.figure(figsize=(6, 5), dpi=200)
            shap.dependence_plot(feat, shap_values, shap_sample, show=False)
            plt.title(f"SHAP Dependence — {feat}")
            plt.tight_layout()
            safe_feat = feat.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(EXTRA_DIR, f"shap_dependence_{safe_feat}.png"), dpi=300, bbox_inches="tight")
            plt.close()
        print("Saved SHAP bar plot + dependence plots for:", top6_features)
    except Exception as e:
        print("[WARN] SHAP dependence plots failed:", e)
else:
    print("\n[B7] SHAP not available, skipped.")

# ------------------------------------------------------------
# B8. Correlation heatmap — top 20 features by importance
# ------------------------------------------------------------
print("\n[B8] Feature correlation heatmap (top 20)...")

top20_feats = fi.head(20)["feature"].tolist()
corr_mat = X_test[top20_feats].corr()

plt.figure(figsize=(11, 9), dpi=200)
im = plt.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson Correlation")
plt.xticks(range(len(top20_feats)), top20_feats, rotation=90, fontsize=8)
plt.yticks(range(len(top20_feats)), top20_feats, fontsize=8)
plt.title("Correlation Heatmap — Top 20 Predictive Features")
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "feature_correlation_heatmap_top20.png"), dpi=300)
plt.close()
corr_mat.to_csv(os.path.join(EXTRA_DIR, "feature_correlation_top20.csv"))
print("Saved correlation heatmap + CSV.")

# ------------------------------------------------------------
# B9. Calibration metrics — slope/intercept + Hosmer-Lemeshow
# ------------------------------------------------------------
print("\n[B9] Calibration slope/intercept + Hosmer-Lemeshow test...")

def calibration_slope_intercept(y_true, p):
    eps = 1e-6
    p = np.clip(np.asarray(p), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(logit_p, y_true)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return slope, intercept

def hosmer_lemeshow(y_true, p, n_bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    df_hl = pd.DataFrame({"y": y_true, "p": p})
    df_hl["decile"] = pd.qcut(df_hl["p"], n_bins, duplicates="drop")
    obs = df_hl.groupby("decile", observed=True)["y"].agg(["sum", "count"])
    obs_events = obs["sum"].values
    n_total = obs["count"].values
    exp_events = df_hl.groupby("decile", observed=True)["p"].sum().values
    exp_nonevents = n_total - exp_events
    obs_nonevents = n_total - obs_events
    stat = np.sum(
        (obs_events - exp_events) ** 2 / (exp_events + 1e-8)
        + (obs_nonevents - exp_nonevents) ** 2 / (exp_nonevents + 1e-8)
    )
    dof = len(obs) - 2
    p_value = 1 - chi2.cdf(stat, dof) if dof > 0 else np.nan
    return float(stat), int(dof), float(p_value)

calib_rows = []
for name, pred in plot_preds_full.items():
    slope, intercept = calibration_slope_intercept(y_test, pred)
    hl_stat, hl_dof, hl_p = hosmer_lemeshow(y_test, pred)
    calib_rows.append({
        "Model": name,
        "Calibration_Slope": slope,
        "Calibration_Intercept": intercept,
        "HosmerLemeshow_Stat": hl_stat,
        "HosmerLemeshow_dof": hl_dof,
        "HosmerLemeshow_p": hl_p,
        "Brier_Score": float(brier_score_loss(y_test, pred)),
    })

calib_df = pd.DataFrame(calib_rows)
calib_path = os.path.join(EXTRA_DIR, "calibration_metrics_table.csv")
calib_df.to_csv(calib_path, index=False)
print(calib_df)
print("Saved:", calib_path)

# ------------------------------------------------------------
# B10. Radar chart — normalized multi-metric comparison
# ------------------------------------------------------------
print("\n[B10] Radar chart of normalized metrics...")

radar_metrics = ["AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "MCC"]
radar_source = results_df[results_df["Model"].isin(plot_preds_full.keys())].copy()
radar_source = radar_source.set_index("Model")[radar_metrics]

# MCC ranges [-1,1] -> rescale to [0,1] for radar comparability
radar_norm = radar_source.copy()
radar_norm["MCC"] = (radar_norm["MCC"] + 1) / 2

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=200)
for model_name in radar_norm.index:
    values = radar_norm.loc[model_name].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model_name, linewidth=1.5)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_ylim(0, 1)
ax.set_title("Multi-Metric Model Comparison (Normalized)", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_DIR, "radar_chart_model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
radar_norm.to_csv(os.path.join(EXTRA_DIR, "radar_chart_values.csv"))
print("Saved radar chart + values.")

# ------------------------------------------------------------
# B11. Consolidated extra-metrics summary (single citation table)
# ------------------------------------------------------------
print("\n[B11] Building consolidated extra-metrics summary...")

summary_extra = results_df[results_df["Model"].isin(plot_preds_full.keys())][
    ["Model", "N", "Preterm_Rate", "AUROC", "AUROC_95CI", "AUPRC", "F1", "MCC",
     "Sensitivity", "Specificity", "PPV", "NPV", "Brier"]
].merge(calib_df[["Model", "Calibration_Slope", "Calibration_Intercept",
                    "HosmerLemeshow_p"]], on="Model", how="left") \
 .merge(youden_df[["Model", "Optimal_Threshold", "Youden_J"]], on="Model", how="left")

summary_path = os.path.join(EXTRA_DIR, "consolidated_metrics_for_manuscript.csv")
summary_extra.to_csv(summary_path, index=False)
print(summary_extra)
print("Saved:", summary_path)

print("\n" + "=" * 90)
print("PART B FINISHED — Extra assets saved to:", EXTRA_DIR)
print("New files:")
print(" - table1_baseline_characteristics.csv")
print(" - auroc_forest_plot.png / auroc_forest_table.csv")
print(" - confusion_matrices_grid.png")
print(" - youden_operating_point_table.csv")
print(" - sens_spec_vs_threshold_best_model.png")
print(" - predicted_probability_distribution.png")
print(" - shap_bar_top20.png / shap_dependence_<feature>.png")
print(" - feature_correlation_heatmap_top20.png / .csv")
print(" - calibration_metrics_table.csv")
print(" - radar_chart_model_comparison.png / radar_chart_values.csv")
print(" - consolidated_metrics_for_manuscript.csv")
print("=" * 90)