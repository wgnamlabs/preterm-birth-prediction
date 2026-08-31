# Figures

This folder is where the **actual PNG outputs from your Kaggle run** go. They are not
included in this scaffold because they live in your Kaggle output zip
(`preterm_results_*.zip`), not in the notebook file itself.

Copy these files here from `/kaggle/working/preterm_scopus_publication_outputs/`
(Part A) and `.../extra_publication_assets/` (Part B):

**From Part A**
- `roc_curves_full_test.png`
- `pr_curves_full_test.png`
- `calibration_full_test.png`
- `cnn_roc_curves_test_subset.png`
- `cnn_pr_curves_test_subset.png`
- `xgboost_feature_importance.png`
- `xgboost_shap_summary.png`
- `decision_curve_analysis.png`
- `cnn_validation_auc_history.png`

**From Part B (`extra_publication_assets/`)**
- `auroc_forest_plot.png`
- `confusion_matrices_grid.png`
- `sens_spec_vs_threshold_best_model.png`
- `predicted_probability_distribution.png`
- `shap_bar_top20.png`
- `feature_correlation_heatmap_top20.png`
- `radar_chart_model_comparison.png`

Once copied, the `![...]` image links in the main `README.md` will render correctly
on GitHub (they already point at these exact filenames).
