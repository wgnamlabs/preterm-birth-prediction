# Scopus-Ready Experimental Report: Preterm Birth Prediction

## Study Design

This pipeline evaluates preterm birth prediction using a hybrid experimental design combining classical tabular machine learning and tabular-to-image feature-map convolutional neural networks. The design emphasizes large-scale validation, leakage prevention, calibration, subgroup analysis, decision-curve analysis, and interpretable feature importance.

## Data and Splitting

- Dataset source: `poojamaheria/natality-2021-to-2023-v2`
- Training set: 7,603,761
- Validation set: 1,629,378
- Independent test set: 1,629,378
- Training preterm rate: 0.1221
- Test preterm rate: 0.1221

## Leakage Prevention

The following target-derived, postnatal, delivery, or outcome-related variables were excluded prior to model training:

- `preterm`
- `weeks_gestational`
- `category_gestation`
- `low_birth_weight`
- `birth_weight_category`
- `APGAR_5_outcome`
- `APGAR_5_category`
- `Low_APGAR`
- `NICU_admission`
- `infant_living`
- `assisted_ventilation`
- `morbidity_none_1`
- `Total_births`
- `Delivery_method`
- `induce_labor`
- `augmentation_labor`
- `presentation`
- `congenital_anomaly_none_1`

## Models

### Full-test tabular ML models
- XGBoost
- LightGBM
- CatBoost
- Logistic stacked ensemble

### Feature-map CNN models
- ResNet18-DeepInsight feature map
- EfficientNetB0-Fingerprint feature map
- SimpleCNN-UMAP highlight map

CNNs use an AMP-safe focal loss with logits; final probabilities are obtained via sigmoid at inference.

## CNN Runtime Caps

- CNN training rows: 120,000
- CNN validation rows: 30,000
- CNN test rows: 120,000
- CNN_TEST_MAX: 120000

When CNN_TEST_MAX is capped, CNN results are reported as subset performance and are not directly stacked with full-test ML predictions.

## Final Model Performance

| Model                          | Eval_Scope      |       N |   Preterm_Rate |    AUROC | AUROC_95CI       |    AUPRC |       F1 |      MCC |   Sensitivity |   Specificity |      PPV |      NPV |     Brier |   Threshold |     TP |      TN |     FP |    FN |
|:-------------------------------|:----------------|--------:|---------------:|---------:|:-----------------|---------:|---------:|---------:|--------------:|--------------:|---------:|---------:|----------:|------------:|-------:|--------:|-------:|------:|
| CNN ResNet18-DeepInsight       | cnn_test_subset |  120000 |       0.122083 | 0.729841 | [0.7263, 0.7331] | 0.374919 | 0.33232  | 0.228644 |      0.641911 |      0.691106 | 0.224193 | 0.93279  | 0.0928038 |    0.116338 |   9404 |   72808 |  32542 |  5246 |
| CNN EfficientNetB0-Fingerprint | cnn_test_subset |  120000 |       0.122083 | 0.728986 | [0.7252, 0.7324] | 0.375375 | 0.3353   | 0.231308 |      0.628396 |      0.705211 | 0.228652 | 0.931726 | 0.0927713 |    0.122582 |   9206 |   74294 |  31056 |  5444 |
| CNN Simple-UMAP                | cnn_test_subset |  120000 |       0.122083 | 0.640181 | [0.6354, 0.6445] | 0.219394 | 0.268287 | 0.131876 |      0.548669 |      0.646578 | 0.177553 | 0.91152  | 0.103087  |    0.120951 |   8038 |   68117 |  37233 |  6612 |
| ML Stacked Ensemble            | full_test       | 1629378 |       0.122086 | 0.741076 | [0.7376, 0.7446] | 0.391476 | 0.348734 | 0.24734  |      0.613747 |      0.73493  | 0.243564 | 0.931891 | 0.0924934 |    0.130571 | 122089 | 1051283 | 379171 | 76835 |
| XGBoost                        | full_test       | 1629378 |       0.122086 | 0.741015 | [0.7376, 0.7445] | 0.391609 | 0.349591 | 0.248173 |      0.610746 |      0.7381   | 0.24488  | 0.931673 | 0.194757  |    0.507473 | 121492 | 1055818 | 374636 | 77432 |
| CatBoost                       | full_test       | 1629378 |       0.122086 | 0.740868 | [0.7374, 0.7443] | 0.390786 | 0.350073 | 0.248532 |      0.60772  |      0.740753 | 0.245846 | 0.931408 | 0.195157  |    0.509762 | 120890 | 1059613 | 370841 | 78034 |
| LightGBM                       | full_test       | 1629378 |       0.122086 | 0.728642 | [0.7257, 0.7322] | 0.361495 | 0.334517 | 0.229628 |      0.621051 |      0.709069 | 0.228906 | 0.930821 | 0.105289  |    0.157276 | 123542 | 1014290 | 416164 | 75382 |

## Statistical Comparison

Pairwise AUROC differences were estimated using bootstrap resampling on a stratified/random sampled subset of the independent test cohort to avoid computational infeasibility of full-test DeLong computation.

| Comparison                      |   Sample_N |   AUROC_Best |   AUROC_Other |   AUROC_Diff |   Bootstrap_CI_Low |   Bootstrap_CI_High |   Approx_p_value | Significant_0.05   |
|:--------------------------------|-----------:|-------------:|--------------:|-------------:|-------------------:|--------------------:|-----------------:|:-------------------|
| ML Stacked Ensemble vs XGBoost  |     200000 |     0.740933 |      0.740881 |  5.17387e-05 |       -9.03585e-05 |         0.000187744 |         0.413333 | False              |
| ML Stacked Ensemble vs LightGBM |     200000 |     0.740933 |      0.729098 |  0.0118353   |        0.0107229   |         0.0131079   |         0        | True               |
| ML Stacked Ensemble vs CatBoost |     200000 |     0.740933 |      0.740715 |  0.00021774  |        7.35343e-05 |         0.000351017 |         0        | True               |

## Top XGBoost Features

| feature                           |   importance |
|:----------------------------------|-------------:|
| plurality                         |  0.524165    |
| Gestational_Hypertension          |  0.0807286   |
| Cigs_3rd_Trimester                |  0.0646439   |
| Pre_pregnancy_Hypertension        |  0.0624196   |
| number_prenatal_visits            |  0.054255    |
| Any_Smoking                       |  0.0425669   |
| Hypertension_Eclampsia            |  0.0289781   |
| Pre_pregnancy_Diabetes            |  0.0254282   |
| Cigs_2nd_Trimester                |  0.0178206   |
| marital_status                    |  0.0147654   |
| Cigs_1st_Trimester                |  0.0105618   |
| Gestational_Diabetes              |  0.00891112  |
| payment_recode                    |  0.00827396  |
| Mother_education                  |  0.00751664  |
| Mother_Age                        |  0.0073867   |
| Parity                            |  0.0062703   |
| Cigs_Before_Pregnancy             |  0.00618529  |
| Mother_Race                       |  0.00556757  |
| infant_sex                        |  0.00498962  |
| prior_dead_recode                 |  0.00474527  |
| mother_nativity                   |  0.00370465  |
| infection_during_pregnancy_none_1 |  0.00351868  |
| Mother_BMI                        |  0.00253667  |
| WIC_participation                 |  0.00240562  |
| Birth_Month                       |  0.000954484 |
| Birth_Year                        |  0.000700403 |

## Subgroup Analysis Preview

| Subgroup                    | Model               |       N |   Preterm_Rate |    AUROC |    AUPRC |     Brier |
|:----------------------------|:--------------------|--------:|---------------:|---------:|---------:|----------:|
| Singleton                   | XGBoost             |    1222 |       0.954992 | 0.614489 | 0.967341 | 0.0440573 |
| Singleton                   | LightGBM            |    1222 |       0.954992 | 0.617683 | 0.968534 | 0.633669  |
| Singleton                   | CatBoost            |    1222 |       0.954992 | 0.601745 | 0.966677 | 0.0429801 |
| Singleton                   | ML Stacked Ensemble |    1222 |       0.954992 | 0.606045 | 0.964367 | 0.147798  |
| Multiple pregnancy          | XGBoost             |   49702 |       0.586858 | 0.641    | 0.727689 | 0.334096  |
| Multiple pregnancy          | LightGBM            |   49702 |       0.586858 | 0.625093 | 0.707141 | 0.402687  |
| Multiple pregnancy          | CatBoost            |   49702 |       0.586858 | 0.63945  | 0.726628 | 0.333329  |
| Multiple pregnancy          | ML Stacked Ensemble |   49702 |       0.586858 | 0.640755 | 0.727674 | 0.235233  |
| Mother age < 20             | XGBoost             | 1629378 |       0.122086 | 0.741015 | 0.391609 | 0.194757  |
| Mother age < 20             | LightGBM            | 1629378 |       0.122086 | 0.728642 | 0.361495 | 0.105289  |
| Mother age < 20             | CatBoost            | 1629378 |       0.122086 | 0.740868 | 0.390786 | 0.195157  |
| Mother age < 20             | ML Stacked Ensemble | 1629378 |       0.122086 | 0.741076 | 0.391476 | 0.0924934 |
| Non-smoking                 | XGBoost             | 1554824 |       0.115376 | 0.729468 | 0.337182 | 0.192383  |
| Non-smoking                 | LightGBM            | 1554824 |       0.115376 | 0.717206 | 0.305537 | 0.100991  |
| Non-smoking                 | CatBoost            | 1554824 |       0.115376 | 0.729299 | 0.336099 | 0.192701  |
| Non-smoking                 | ML Stacked Ensemble | 1554824 |       0.115376 | 0.729508 | 0.337104 | 0.0906662 |
| Smoking                     | XGBoost             |   74554 |       0.262025 | 0.824367 | 0.744786 | 0.244264  |
| Smoking                     | LightGBM            |   74554 |       0.262025 | 0.802679 | 0.710542 | 0.194907  |
| Smoking                     | CatBoost            |   74554 |       0.262025 | 0.824161 | 0.741683 | 0.246377  |
| Smoking                     | ML Stacked Ensemble |   74554 |       0.262025 | 0.824611 | 0.743364 | 0.130601  |
| No gestational hypertension | XGBoost             | 1471546 |       0.111531 | 0.732838 | 0.368779 | 0.182275  |
| No gestational hypertension | LightGBM            | 1471546 |       0.111531 | 0.719984 | 0.33939  | 0.0981297 |
| No gestational hypertension | CatBoost            | 1471546 |       0.111531 | 0.732693 | 0.367914 | 0.182653  |
| No gestational hypertension | ML Stacked Ensemble | 1471546 |       0.111531 | 0.732902 | 0.36858  | 0.0862248 |
| Gestational hypertension    | XGBoost             |  157832 |       0.220494 | 0.709273 | 0.477006 | 0.311135  |
| Gestational hypertension    | LightGBM            |  157832 |       0.220494 | 0.691507 | 0.424332 | 0.172034  |
| Gestational hypertension    | CatBoost            |  157832 |       0.220494 | 0.709099 | 0.476598 | 0.311733  |
| Gestational hypertension    | ML Stacked Ensemble |  157832 |       0.220494 | 0.709434 | 0.477218 | 0.150939  |
| BMI < 18.5                  | XGBoost             |   43000 |       0.139442 | 0.7246   | 0.38488  | 0.220834  |
| BMI < 18.5                  | LightGBM            |   43000 |       0.139442 | 0.712352 | 0.358465 | 0.117467  |
| BMI < 18.5                  | CatBoost            |   43000 |       0.139442 | 0.724771 | 0.38529  | 0.22078   |
| BMI < 18.5                  | ML Stacked Ensemble |   43000 |       0.139442 | 0.724944 | 0.385488 | 0.105711  |
| BMI 18.5-29.9               | XGBoost             | 1049489 |       0.111769 | 0.738293 | 0.369619 | 0.182133  |
| BMI 18.5-29.9               | LightGBM            | 1049489 |       0.111769 | 0.727122 | 0.340904 | 0.098235  |
| BMI 18.5-29.9               | CatBoost            | 1049489 |       0.111769 | 0.738168 | 0.368641 | 0.182539  |
| BMI 18.5-29.9               | ML Stacked Ensemble | 1049489 |       0.111769 | 0.738357 | 0.369403 | 0.0863184 |
| BMI >= 30                   | XGBoost             |  536889 |       0.140863 | 0.740851 | 0.424557 | 0.217346  |
| BMI >= 30                   | LightGBM            |  536889 |       0.140863 | 0.726666 | 0.392725 | 0.118101  |
| BMI >= 30                   | CatBoost            |  536889 |       0.140863 | 0.740644 | 0.423886 | 0.217769  |
| BMI >= 30                   | ML Stacked Ensemble |  536889 |       0.140863 | 0.740894 | 0.424477 | 0.103506  |

## Decision Curve Analysis

Decision curve analysis was generated across threshold probabilities from 0.01 to 0.50. See `decision_curve_analysis.csv` and `decision_curve_analysis.png`.

## Recommended Manuscript Framing

If feature-map CNNs do not outperform tree-based ensembles, the manuscript should be framed as a large-scale benchmark and feasibility study of tabular-to-image representation learning for preterm birth prediction, rather than as a superiority claim for CNNs.
