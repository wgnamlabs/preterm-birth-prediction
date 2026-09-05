# Methodology

## 1. Data source

- Dataset: [`poojamaheria/natality-2021-to-2023-v2`](https://www.kaggle.com/datasets/poojamaheria/natality-2021-to-2023-v2)
  on Kaggle — derived from U.S. CDC/NCHS Natality (birth certificate) public-use files, 2021–2023.
- Raw rows after filtering to the two gestation categories used here: several million;
  70/15/15 stratified split leaves an **independent test set of 1,629,378 records**
  (never touched during training or hyperparameter selection).
- Preterm prevalence in the data: **12.21%** (`preterm = 1` when `category_gestation == 1`,
  i.e. < 37 completed weeks; `category_gestation == 2`, term, is the reference class).

## 2. Leakage removal

Any field that is only known **after** the birth outcome is determined is dropped
before any modeling step, because such fields would leak the label:

```
preterm, weeks_gestational, category_gestation, low_birth_weight,
birth_weight_category, APGAR_5_outcome, APGAR_5_category, Low_APGAR,
NICU_admission, infant_living, assisted_ventilation, morbidity_none_1,
morbidity_0, Total_births, Delivery_method, induce_labor,
augmentation_labor, presentation, congenital_anomaly_none_1
```

Only variables available at (or before) delivery admission — maternal
demographics, prior pregnancy history, prenatal care utilization, comorbidities,
smoking, plurality, etc. — remain as predictors.

## 3. Preprocessing

- ID-like columns (patient/record/certificate identifiers, unnamed index columns)
  are dropped.
- Categorical (object/category dtype) columns are ordinal-encoded
  (`sklearn.preprocessing.OrdinalEncoder`, unknown categories mapped to -1).
- Columns with >50% missing values are dropped; remaining missing numeric values
  are median-imputed.
- Constant columns are dropped.
- All features cast to `float32`.

## 4. Train / validation / test split

Stratified by label, fixed `random_state=42`:

- Train: 70%
- Validation: 15% (used for early stopping / meta-learner fitting / threshold calibration)
- Test: 15%, held out and **never used for any fitting decision**

## 5. Tabular models

Three gradient-boosted tree models, each class-weighted for the ~12% positive rate
via `scale_pos_weight` (XGBoost/LightGBM) or `class_weights` (CatBoost):

| Model | Key hyperparameters |
|---|---|
| XGBoost | 900 trees, depth 6, lr 0.035, subsample/colsample 0.85, `hist` tree method, GPU if available |
| LightGBM | 1200 trees (early stopping patience 100), 63 leaves, lr 0.03, CPU (more stable than GPU for this shape) |
| CatBoost | 900 iterations, depth 7, lr 0.035, GPU if available |

**ML Stacked Ensemble**: out-of-fold-style stacking — the three base models'
*validation-set* predictions train a logistic-regression meta-learner
(`C=0.5`), which is then applied to the base models' *test-set* predictions.
This is the best-performing model overall (AUROC 0.7411).

## 6. Tabular-to-image CNN branch (ablation)

Each row is also encoded into a fixed-size image so a CNN can be trained on it,
via a custom `FeatureMapTransformer`:

- **DeepInsight-style (1-channel)**: each feature gets a 2D position (from PCA on
  a 1 − |correlation| distance matrix between features), and a Gaussian "blob"
  proportional to the (min-max scaled) feature value is stamped at that position.
- **Fingerprint (3-channel)**: at each feature's position, stamps
  (a) positive deviation from the feature's median, (b) negative deviation,
  (c) raw scaled value — into separate R/G/B channels.
- **UMAP-density highlight (1-channel)**: a 2D UMAP (or PCA fallback) embedding of
  the sample gives a per-row density map blended with a Gaussian highlight at the
  sample's own embedded position.

CNN backbones: `ResNet18` (1-channel input, custom head) and `EfficientNetB0`
(3-channel input, custom head), trained with a focal loss
(`alpha=0.75, gamma=2.0`) under automatic mixed precision, up to 18 epochs with
early stopping (patience 5), plus light augmentation (flips, small Gaussian noise).
A fallback plain CNN (`SimpleCNN`) is used for the UMAP-highlight branch and if
`torchvision` is unavailable.

**Why capped?** GPU wall-clock time on Kaggle's T4 makes training + inference on
the full 11M+/1.6M train/test rows impractical for three separate CNN branches, so
CNN train/val/test are stratified-capped to 120K/30K/120K rows respectively
(`CNN_TRAIN_MAX`, `CNN_VAL_MAX`, `CNN_TEST_MAX`). This is why CNN metrics are
reported on `cnn_test_subset` scope, not `full_test`, and are not directly
comparable sample-for-sample to the tabular models.

CNN test-set probabilities are Platt-calibrated (logistic regression fit on
validation predictions) before evaluation.

## 7. Hybrid stacking (conditional)

A "Hybrid ML+CNN Stack" (logistic-regression meta-learner over all base model +
CNN predictions) is only built when `CNN_TEST_MAX = None` (i.e. CNN evaluated on
the full test set) — otherwise mixing full-test ML predictions with subset-only
CNN predictions in one reported number would be statistically misleading, so it's
skipped and logged as such. The run reported in this README used the capped
setting, so no hybrid number is reported.

## 8. Evaluation

For every model, on its respective test scope:

- **AUROC** with a 95% bootstrap confidence interval (100 resamples, capped at
  200K rows per resample for tractability on the 1.6M-row full test set).
- **AUPRC** (average precision).
- **Youden-optimal threshold** (`argmax(TPR − FPR)` on the ROC curve) used to binarize
  predictions for F1, MCC, sensitivity, specificity, PPV, NPV, and the confusion matrix.
- **Brier score** for probabilistic calibration.

Additional publication-style assets (Part B):

- **Table 1** baseline characteristics by outcome group.
- **Forest plot** of AUROC + 95% CI across models.
- **Confusion matrices** grid at each model's optimal threshold.
- **Sensitivity/specificity vs. threshold** curve for the best model.
- **Predicted probability distribution** by true class.
- **SHAP** global bar plot (top 20 features) + dependence plots for top features
  (best tabular model).
- **Feature correlation heatmap** (top 20 features by SHAP importance).
- **Calibration metrics table**: calibration slope/intercept, Hosmer-Lemeshow
  statistic and p-value, Brier score, per model.
- **Radar chart** of normalized multi-metric comparison across models.
- **Decision curve analysis** (net benefit vs. treat-all/treat-none across
  threshold probabilities).
- **Subgroup AUROC** analysis by smoking status, BMI category, and plurality.

## 9. Reproducibility

- Global seed `SEED = 42` for `random`, `numpy`, `torch` (+ CUDA).
- All Kaggle-dataset downloads, model configs, and output paths are controlled by
  the config block at the top of
  [`notebooks/part_a_main_pipeline.ipynb`](../notebooks/part_a_main_pipeline.ipynb)
  (`RUN_CNN`, `RUN_SHAP`, `RUN_SUBGROUP`, `RUN_DCA`, `CNN_*_MAX`,
  `BOOTSTRAP_CI_*`, `SHAP_SAMPLE_N`, etc.) — flip these to reproduce faster/slower
  or full-scale runs.
