"""
Part A - Main Pipeline (Tabular ML + Tabular-to-Image CNN)

Extracted from cdc-final-16-tri-u-ca.ipynb for readability / code review.
Run order: part_a -> part_b -> part_c (each reuses variables from the previous
one, so they must be executed in the same Python session / notebook kernel).
"""

# ============================================================
# PRETERM BIRTH PREDICTION - SCOPUS-READY EXPERIMENTAL PIPELINE
# Hybrid Tabular ML + Tabular-to-Image Feature Map CNN
# Optimized for Kaggle GPU Tesla T4
# ============================================================
#
# Key design goals:
# 1) Full-test evaluation for tabular ML models on all held-out samples.
# 2) Tabular-to-image CNN branches using feature maps:
#    - DeepInsight-like 1-channel feature map
#    - Fingerprint 3-channel feature map
#    - UMAP-density highlight 1-channel feature map
# 3) AMP-safe CNN training: models output logits; loss uses BCEWithLogits.
# 4) Publication outputs:
#    - final metrics CSV
#    - ROC / PR / Calibration plots
#    - SHAP summary
#    - subgroup analysis
#    - decision curve analysis
#    - fast bootstrap model comparison instead of slow full DeLong
#    - Markdown report
# 5) Avoids notebook hanging on full-test DeLong with 1.6M+ samples.
#
# Recommended Kaggle settings:
# - Accelerator: GPU T4
# - Internet: ON
# - Run as a single notebook cell or upload as .py and run.
# ============================================================

# ============================================================
# 0. INSTALLS
# ============================================================

import os
import sys
import subprocess


def pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"[WARN] Could not install {pkg}: {e}")


for _pkg in [
    "kaggle",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "umap-learn",
    "imbalanced-learn",
    "tabulate",
]:
    try:
        __import__(_pkg.replace("-", "_"))
    except Exception:
        pip_install(_pkg)

# ============================================================
# 1. IMPORTS AND CONFIG
# ============================================================

import gc
import time
import zipfile
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from scipy.ndimage import gaussian_filter
from scipy import stats

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    import umap as umap_lib
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except Exception:
    HAS_AMP = False


# -------------------- User-adjustable config --------------------
SEED = 42

# Kaggle dataset used in your current experiment.
DATASET_REF = "poojamaheria/natality-2021-to-2023-v2"
ZIP_PATH = "/kaggle/working/natality-2021-to-2023-v2.zip"
EXTRACT_DIR = "/kaggle/working/natality_2023"

OUT_DIR = "/kaggle/working/preterm_scopus_publication_outputs" if os.path.exists("/kaggle") else "./preterm_scopus_publication_outputs"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Switches
RUN_CNN = True
RUN_SHAP = True
RUN_SUBGROUP = True
RUN_DCA = True
RUN_FAST_MODEL_COMPARISON = True

# CNN feature-map settings.
IMG_SIZE = 64
BATCH_SIZE = 128
CNN_EPOCHS = 18
PATIENCE = 5
USE_AMP = True

# CNN caps keep T4 runtime manageable.
# ML models still evaluate on the full independent test set.
CNN_TRAIN_MAX = 120_000
CNN_VAL_MAX = 30_000
CNN_TEST_MAX = 120_000
CNN_PRED_CHUNK = 20_000

# If you want full-test CNN and full hybrid stack, set CNN_TEST_MAX = None.
# Warning: it can be slow and memory-heavy.

# Statistical computation caps.
BOOTSTRAP_CI_REPEATS = 100
BOOTSTRAP_CI_MAX_N = 200_000
MODEL_COMPARE_SAMPLE_N = 200_000
MODEL_COMPARE_BOOTSTRAPS = 300

# SHAP sample.
SHAP_SAMPLE_N = 5_000

# LightGBM sometimes underperforms on GPU/early-stopping in Kaggle builds.
# CPU is often more stable for 26 tabular features and millions of rows.
LGB_USE_GPU = False

# ---------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 90)
print("PRETERM BIRTH SCOPUS-READY PIPELINE")
print("Hybrid: Tabular ML + Tabular-to-Image Feature Map CNN")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
print("Output:", OUT_DIR)
print("=" * 90)

# ============================================================
# 2. DATA DOWNLOAD AND LOAD
# ============================================================

print("\n[1/18] Downloading Kaggle dataset...")
Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

# Kaggle command. If already available, this will overwrite the zip.
cmd = ["kaggle", "datasets", "download", "-d", DATASET_REF, "-p", "/kaggle/working", "--force"]
try:
    subprocess.check_call(cmd)
except Exception as e:
    print("[WARN] Kaggle download failed. If data is already present, continuing.")
    print(e)

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"Missing zip file: {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

csv_files = []
for root, _, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.lower().endswith(".csv"):
            csv_files.append(os.path.join(root, f))

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found after extraction.")

CSV_PATH = None
for f in csv_files:
    if "(1)" in f:
        CSV_PATH = f
        break
if CSV_PATH is None:
    CSV_PATH = csv_files[0]

print("Using:", CSV_PATH)

print("\n[2/18] Loading full data...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print("Raw shape:", df.shape)

if "category_gestation" not in df.columns:
    raise ValueError("Missing target source column: category_gestation")

# The current dataset encodes gestation category; use category 1 as preterm and 2 as term.
df = df[df["category_gestation"].isin([1, 2])].copy()
df["preterm"] = (df["category_gestation"] == 1).astype(np.int8)

print("Cleaned shape:", df.shape)
print("Preterm rate:", float(df["preterm"].mean()))
print(df["preterm"].value_counts())

# ============================================================
# 3. LEAKAGE REMOVAL AND PREPROCESSING
# ============================================================

print("\n[3/18] Removing leakage columns...")

LEAKAGE_COLS = [
    "preterm",
    "weeks_gestational",
    "category_gestation",
    "low_birth_weight",
    "birth_weight_category",
    "APGAR_5_outcome",
    "APGAR_5_category",
    "Low_APGAR",
    "NICU_admission",
    "infant_living",
    "assisted_ventilation",
    "morbidity_none_1",
    "morbidity_0",
    "Total_births",
    "Delivery_method",
    "induce_labor",
    "augmentation_labor",
    "presentation",
    "congenital_anomaly_none_1",
]
LEAKAGE_COLS = [c for c in LEAKAGE_COLS if c in df.columns]

X = df.drop(columns=LEAKAGE_COLS, errors="ignore")
y = df["preterm"].astype(np.int8)

del df
gc.collect()

print("Dropped leakage columns:")
for c in LEAKAGE_COLS:
    print(" -", c)
print("Feature shape before preprocess:", X.shape)

print("\n[4/18] Preprocessing tabular features...")


def preprocess_full(X_in: pd.DataFrame):
    X_out = X_in.copy()

    id_patterns = [
        "id", "cpf", "patient", "index", "unnamed", "record", "certificate", "file", "row"
    ]
    drop_id = [c for c in X_out.columns if any(p in c.lower() for p in id_patterns)]
    if drop_id:
        X_out = X_out.drop(columns=drop_id, errors="ignore")
        print("Dropped ID-like columns:", len(drop_id))

    cat_cols = []
    for c in X_out.columns:
        if X_out[c].dtype == "object" or str(X_out[c].dtype) == "category":
            cat_cols.append(c)

    for c in cat_cols:
        X_out[c] = X_out[c].astype(str).fillna("missing")

    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_out[cat_cols] = enc.fit_transform(X_out[cat_cols]).astype(np.float32)

    drop_high_missing = []
    for c in list(X_out.columns):
        X_out[c] = pd.to_numeric(X_out[c], errors="coerce")
        miss = X_out[c].isna().mean()
        if miss > 0.50:
            drop_high_missing.append(c)
        else:
            med = X_out[c].median()
            if pd.isna(med):
                med = 0
            X_out[c] = X_out[c].fillna(med).astype(np.float32)

    if drop_high_missing:
        X_out = X_out.drop(columns=drop_high_missing, errors="ignore")
        print("Dropped high-missing columns:", len(drop_high_missing))

    nunique = X_out.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X_out = X_out.drop(columns=const_cols, errors="ignore")
        print("Dropped constant columns:", len(const_cols))

    X_out = X_out.replace([np.inf, -np.inf], 0).fillna(0)
    return X_out.astype(np.float32)


X = preprocess_full(X)
print("After preprocess:", X.shape)
print("Columns:", X.columns.tolist())

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[5/18] Train / validation / test split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

del X, y, X_temp, y_temp
gc.collect()

print("Train:", X_train.shape, "rate:", float(y_train.mean()))
print("Val:  ", X_val.shape, "rate:", float(y_val.mean()))
print("Test: ", X_test.shape, "rate:", float(y_test.mean()))

# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

print("\n[6/18] Defining evaluation helpers...")


def optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def bootstrap_auc_ci(y_true, y_score, n_boot=BOOTSTRAP_CI_REPEATS, max_n=BOOTSTRAP_CI_MAX_N):
    rng = np.random.RandomState(SEED)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) > max_n:
        idx = rng.choice(len(y_true), max_n, replace=False)
        y_true = y_true[idx]
        y_score = y_score[idx]

    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    if not aucs:
        return "[N/A]"

    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return f"[{lo:.4f}, {hi:.4f}]"


def evaluate_model(y_true, y_proba, name, eval_scope="full_test"):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    auc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    ci = bootstrap_auc_ci(y_true, y_proba)

    thresh = optimal_threshold(y_true, y_proba)
    y_pred = (y_proba >= thresh).astype(int)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)

    print("\n" + "=" * 76)
    print(name)
    print("Scope:", eval_scope)
    print("AUROC:", round(auc, 6), ci)
    print("AUPRC:", round(auprc, 6))
    print("F1:", round(f1, 6), "MCC:", round(mcc, 6))
    print("Sensitivity:", round(sens, 6), "Specificity:", round(spec, 6))
    print("PPV:", round(ppv, 6), "NPV:", round(npv, 6))
    print("Brier:", round(brier, 6), "Threshold:", round(thresh, 6))
    print("=" * 76)

    return {
        "Model": name,
        "Eval_Scope": eval_scope,
        "N": int(len(y_true)),
        "Preterm_Rate": float(np.mean(y_true)),
        "AUROC": float(auc),
        "AUROC_95CI": ci,
        "AUPRC": float(auprc),
        "F1": float(f1),
        "MCC": float(mcc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Brier": float(brier),
        "Threshold": float(thresh),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def platt_calibrate(y_cal, p_cal, p_target):
    try:
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        model.fit(np.asarray(p_cal).reshape(-1, 1), np.asarray(y_cal))
        return model.predict_proba(np.asarray(p_target).reshape(-1, 1))[:, 1]
    except Exception as e:
        print("[WARN] Platt calibration failed:", e)
        return p_target


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 6. TABULAR ML MODELS
# ============================================================

print("\n[7/18] Training tabular ML models...")

results = []
val_preds = {}
test_preds = {}
model_objects = {}

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)
print("scale_pos_weight:", scale_pos_weight)

# -------------------- XGBoost --------------------
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=900,
    learning_rate=0.035,
    max_depth=6,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    reg_alpha=0.2,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    device="cuda" if torch.cuda.is_available() else "cpu",
    random_state=SEED,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
val_preds["XGBoost"] = xgb_model.predict_proba(X_val)[:, 1]
test_preds["XGBoost"] = xgb_model.predict_proba(X_test)[:, 1]
model_objects["XGBoost"] = xgb_model
results.append(evaluate_model(y_test, test_preds["XGBoost"], "XGBoost", "full_test"))

# -------------------- LightGBM --------------------
print("\nTraining LightGBM...")
lgb_params = dict(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    reg_alpha=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

if LGB_USE_GPU and torch.cuda.is_available():
    lgb_params.update({"device": "gpu"})

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
)
val_preds["LightGBM"] = lgb_model.predict_proba(X_val)[:, 1]
test_preds["LightGBM"] = lgb_model.predict_proba(X_test)[:, 1]
model_objects["LightGBM"] = lgb_model
results.append(evaluate_model(y_test, test_preds["LightGBM"], "LightGBM", "full_test"))

# -------------------- CatBoost --------------------
print("\nTraining CatBoost...")
cat_model = CatBoostClassifier(
    iterations=900,
    learning_rate=0.035,
    depth=7,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    class_weights=[1.0, scale_pos_weight],
    task_type="GPU" if torch.cuda.is_available() else "CPU",
    verbose=100,
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
val_preds["CatBoost"] = cat_model.predict_proba(X_val)[:, 1]
test_preds["CatBoost"] = cat_model.predict_proba(X_test)[:, 1]
model_objects["CatBoost"] = cat_model
results.append(evaluate_model(y_test, test_preds["CatBoost"], "CatBoost", "full_test"))

# -------------------- ML Stacked Ensemble --------------------
print("\nTraining ML Stacked Ensemble...")
ml_names = ["XGBoost", "LightGBM", "CatBoost"]
stack_val = np.column_stack([val_preds[m] for m in ml_names])
stack_test = np.column_stack([test_preds[m] for m in ml_names])

meta_ml = LogisticRegression(C=0.5, max_iter=2000, random_state=SEED)
meta_ml.fit(stack_val, y_val)
val_preds["ML Stacked Ensemble"] = meta_ml.predict_proba(stack_val)[:, 1]
test_preds["ML Stacked Ensemble"] = meta_ml.predict_proba(stack_test)[:, 1]
model_objects["ML Stacked Ensemble"] = meta_ml
results.append(evaluate_model(y_test, test_preds["ML Stacked Ensemble"], "ML Stacked Ensemble", "full_test"))

# ============================================================
# 7. TABULAR-TO-IMAGE FEATURE MAP TRANSFORMER
# ============================================================

print("\n[8/18] Defining tabular-to-image transformer...")


def stratified_cap(X_df, y_ser, max_n):
    y_ser = pd.Series(y_ser).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    if max_n is None or len(X_df) <= max_n:
        return X_df, y_ser

    _, X_sub, _, y_sub = train_test_split(
        X_df,
        y_ser,
        test_size=max_n / len(X_df),
        random_state=SEED,
        stratify=y_ser,
    )
    return X_sub.reset_index(drop=True), pd.Series(y_sub).reset_index(drop=True)


class FeatureMapTransformer:
    """Tabular-to-image transformer using feature-position maps.

    Methods:
    - transform_deepinsight: 1 x H x W map from normalized feature values.
    - transform_fingerprint: 3 x H x W map for above-median, below-median, raw value.
    - transform_umap_highlight: 1 x H x W sample manifold highlight.
    """

    def __init__(self, img_size=64, sigma=2.0):
        self.img_size = img_size
        self.sigma = sigma
        self.rad = int(2 * sigma)
        self.scaler = MinMaxScaler()
        self.medians = None
        self.feature_medians = None
        self.feature_positions = None
        self.pca_reducer = None
        self.umap_reducer = None
        self._coord_min = None
        self._coord_max = None
        self._density = None
        self._kernel = self._build_kernel()
        self.n_features = None

    def _build_kernel(self):
        ax = np.arange(-self.rad, self.rad + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        return np.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma ** 2)).astype(np.float32)

    def fit(self, X, y=None):
        X = X.copy()
        self.medians = X.median()
        Xn = self.scaler.fit_transform(X.fillna(self.medians)).astype(np.float32)
        n, f = Xn.shape
        self.n_features = f
        self.feature_medians = np.median(Xn, axis=0)

        if f > 1:
            corr = np.abs(np.corrcoef(Xn.T))
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            dist = 1 - np.clip(corr, 0, 1)
            pca_feat = PCA(n_components=2, random_state=SEED)
            feat_2d = pca_feat.fit_transform(dist)
        else:
            feat_2d = np.array([[0.5, 0.5]], dtype=np.float32)

        sc = MinMaxScaler(feature_range=(1, self.img_size - 2))
        self.feature_positions = sc.fit_transform(feat_2d).astype(int)
        self.feature_positions = np.clip(self.feature_positions, 0, self.img_size - 1)

        n_comp = min(2, max(1, min(Xn.shape[0], Xn.shape[1]) - 1))
        self.pca_reducer = PCA(n_components=n_comp, random_state=SEED)
        self.pca_reducer.fit(Xn)

        sample_n = min(3000, len(Xn))
        idx = np.random.RandomState(SEED).choice(len(Xn), sample_n, replace=False)

        if HAS_UMAP and sample_n > 20:
            try:
                self.umap_reducer = umap_lib.UMAP(
                    n_components=2,
                    n_neighbors=min(15, sample_n - 1),
                    min_dist=0.1,
                    random_state=SEED,
                    low_memory=True,
                )
                if y is not None:
                    self.umap_reducer.fit(Xn[idx], np.asarray(y)[idx])
                else:
                    self.umap_reducer.fit(Xn[idx])
            except Exception as e:
                print("[WARN] UMAP failed, PCA fallback:", e)
                self.umap_reducer = None

        if self.umap_reducer is not None:
            coords = self.umap_reducer.transform(Xn[idx])
        else:
            coords = self.pca_reducer.transform(Xn[idx])
            if coords.shape[1] == 1:
                coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        self._coord_min = mn
        self._coord_max = np.where(mx - mn < 1e-8, mn + 1, mx)

        norm = ((coords - self._coord_min) / (self._coord_max - self._coord_min) * (self.img_size - 1)).astype(int)
        norm = np.clip(norm, 0, self.img_size - 1)

        density = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for cx, cy in norm:
            density[cy, cx] += 1
        density = gaussian_filter(density, sigma=1.5)
        self._density = density / (density.max() + 1e-8)
        return self

    def _draw_blob(self, img, r, c, val):
        val = float(np.clip(val, 0, 1))
        if val <= 1e-8:
            return img

        h, w = img.shape
        r_min, r_max = max(0, r - self.rad), min(h, r + self.rad + 1)
        c_min, c_max = max(0, c - self.rad), min(w, c + self.rad + 1)

        kr_min = r_min - (r - self.rad)
        kr_max = self._kernel.shape[0] - ((r + self.rad + 1) - r_max)
        kc_min = c_min - (c - self.rad)
        kc_max = self._kernel.shape[1] - ((c + self.rad + 1) - c_max)

        img[r_min:r_max, c_min:c_max] = np.maximum(
            img[r_min:r_max, c_min:c_max],
            val * self._kernel[kr_min:kr_max, kc_min:kc_max],
        )
        return img

    def transform_deepinsight(self, X):
        Xn = self.scaler.transform(X.copy().fillna(self.medians)).astype(np.float32)
        n = len(Xn)
        imgs = np.zeros((n, 1, self.img_size, self.img_size), dtype=np.float32)
        for i in range(n):
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            for j in range(self.n_features):
                r, c = self.feature_positions[j]
                self._draw_blob(img, r, c, Xn[i, j])
            imgs[i, 0] = img
        return imgs

    def transform_fingerprint(self, X):
        Xn = self.scaler.transform(X.copy().fillna(self.medians)).astype(np.float32)
        n = len(Xn)
        imgs = np.zeros((n, 3, self.img_size, self.img_size), dtype=np.float32)
        for i in range(n):
            xi = Xn[i]
            dev = xi - self.feature_medians
            img_r = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            img_g = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            img_b = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            for j in range(self.n_features):
                r, c = self.feature_positions[j]
                self._draw_blob(img_r, r, c, max(0.0, dev[j]))
                self._draw_blob(img_g, r, c, max(0.0, -dev[j]))
                self._draw_blob(img_b, r, c, xi[j])
            imgs[i, 0] = img_r
            imgs[i, 1] = img_g
            imgs[i, 2] = img_b
        return imgs

    def transform_umap_highlight(self, X):
        Xn = self.scaler.transform(X.copy().fillna(self.medians)).astype(np.float32)
        if self.umap_reducer is not None:
            try:
                coords = self.umap_reducer.transform(Xn)
            except Exception:
                coords = self.pca_reducer.transform(Xn)
        else:
            coords = self.pca_reducer.transform(Xn)

        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

        norm = ((coords - self._coord_min) / (self._coord_max - self._coord_min) * (self.img_size - 1)).astype(int)
        norm = np.clip(norm, 0, self.img_size - 1)

        n = len(Xn)
        imgs = np.zeros((n, 1, self.img_size, self.img_size), dtype=np.float32)
        ys, xs = np.mgrid[0:self.img_size, 0:self.img_size]
        sigma = max(3.0, self.img_size * 0.06)
        for i, (cx, cy) in enumerate(norm):
            img = self._density * 0.3
            gauss = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            imgs[i, 0] = np.clip(img + gauss * 0.7, 0, 1)
        return imgs

# ============================================================
# 8. CNN MODELS AND TRAINING
# ============================================================

print("\n[9/18] Defining CNN models...")


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target == 1, prob, 1 - prob)
        alpha_t = torch.where(
            target == 1,
            torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            torch.tensor(1 - self.alpha, device=logits.device, dtype=logits.dtype),
        )
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class FeatureImgDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-2])
            if torch.rand(1).item() > 0.5:
                img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)
        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )
        self.gradcam_layer = self.net[8]

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


class ResNet18Custom(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        base.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.base = base
        self.gradcam_layer = base.layer4[1].conv2

    def forward(self, x):
        return self.base(x).squeeze(-1)


class EfficientNetB0Custom(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        old = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_channels,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=False,
        )
        in_features = base.classifier[-1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 1),
        )
        self.base = base
        self.gradcam_layer = base.features[-1][0]

    def forward(self, x):
        out = self.base(x)
        return out.squeeze(-1) if out.ndim > 1 else out


def train_cnn(model, tr_imgs, vl_imgs, y_tr, y_vl, name):
    pos_ratio = float(np.mean(y_tr))
    alpha = max(0.65, 1 - pos_ratio)
    criterion = FocalLossWithLogits(alpha=alpha, gamma=2.0)

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(3, CNN_EPOCHS))

    train_ds = FeatureImgDataset(tr_imgs, y_tr, augment=True)
    val_ds = FeatureImgDataset(vl_imgs, y_vl, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    amp_enabled = HAS_AMP and USE_AMP and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    best_auc = -1
    best_state = None
    no_imp = 0
    history = []

    print(f"\nTraining CNN: {name}")
    for epoch in range(CNN_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            if amp_enabled:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += float(loss.item())

        scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                prob = torch.sigmoid(logits)
                vp.extend(prob.detach().cpu().numpy())
                vt.extend(labels.numpy())

        vp = np.asarray(vp)
        vt = np.asarray(vt)
        val_auc = roc_auc_score(vt, vp) if len(np.unique(vt)) > 1 else 0.5
        avg_loss = epoch_loss / max(len(train_loader), 1)

        history.append({"epoch": epoch + 1, "loss": avg_loss, "val_auc": val_auc, "Model": name})
        print(f"  Epoch {epoch + 1:02d}/{CNN_EPOCHS} | loss={avg_loss:.5f} | val_auc={val_auc:.5f} | best={max(best_auc, val_auc):.5f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_cnn_in_chunks(model, transformer_func, X_df, chunk_size=CNN_PRED_CHUNK):
    model.eval()
    preds = []
    for start in range(0, len(X_df), chunk_size):
        end = min(start + chunk_size, len(X_df))
        imgs = transformer_func(X_df.iloc[start:end])
        ds = FeatureImgDataset(imgs, np.zeros(len(imgs)), augment=False)
        loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch_imgs, _ in loader:
                batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
                logits = model(batch_imgs)
                prob = torch.sigmoid(logits)
                preds.extend(prob.detach().cpu().numpy())
        del imgs, ds, loader
        free_memory()
    return np.asarray(preds)

# ============================================================
# 9. RUN CNN FEATURE-MAP BRANCHES
# ============================================================

cnn_histories = []
cnn_context = {}

if RUN_CNN:
    print("\n[10/18] Preparing scaled data for CNN feature maps...")

    std_scaler = StandardScaler()
    X_train_sc = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns).astype(np.float32)
    X_val_sc = pd.DataFrame(std_scaler.transform(X_val), columns=X_val.columns).astype(np.float32)
    X_test_sc = pd.DataFrame(std_scaler.transform(X_test), columns=X_test.columns).astype(np.float32)

    X_cnn_train, y_cnn_train = stratified_cap(X_train_sc, y_train, CNN_TRAIN_MAX)
    X_cnn_val, y_cnn_val = stratified_cap(X_val_sc, y_val, CNN_VAL_MAX)
    X_cnn_test, y_cnn_test = stratified_cap(X_test_sc, y_test, CNN_TEST_MAX)

    cnn_context["X_cnn_train_N"] = len(X_cnn_train)
    cnn_context["X_cnn_val_N"] = len(X_cnn_val)
    cnn_context["X_cnn_test_N"] = len(X_cnn_test)
    cnn_context["CNN_TEST_MAX"] = CNN_TEST_MAX

    if CNN_TEST_MAX is not None:
        print(f"CNN test capped to {len(X_cnn_test):,}; CNN metrics are on capped test subset.")
    else:
        print(f"CNN test full size: {len(X_cnn_test):,}")

    print("CNN train:", X_cnn_train.shape, "rate:", float(np.mean(y_cnn_train)))
    print("CNN val:  ", X_cnn_val.shape, "rate:", float(np.mean(y_cnn_val)))
    print("CNN test: ", X_cnn_test.shape, "rate:", float(np.mean(y_cnn_test)))

    print("\nFitting FeatureMapTransformer...")
    fmt = FeatureMapTransformer(img_size=IMG_SIZE, sigma=2.0)
    fmt.fit(X_cnn_train, y_cnn_train)

    cnn_cfgs = []
    if HAS_TORCHVISION:
        cnn_cfgs.append({
            "name": "CNN ResNet18-DeepInsight",
            "tfm": fmt.transform_deepinsight,
            "model": ResNet18Custom(in_channels=1),
        })
        cnn_cfgs.append({
            "name": "CNN EfficientNetB0-Fingerprint",
            "tfm": fmt.transform_fingerprint,
            "model": EfficientNetB0Custom(in_channels=3),
        })
    else:
        cnn_cfgs.append({
            "name": "CNN Simple-DeepInsight",
            "tfm": fmt.transform_deepinsight,
            "model": SimpleCNN(in_channels=1),
        })

    cnn_cfgs.append({
        "name": "CNN Simple-UMAP",
        "tfm": fmt.transform_umap_highlight,
        "model": SimpleCNN(in_channels=1),
    })

    for cfg in cnn_cfgs:
        try:
            print(f"\nCreating feature images for {cfg['name']}...")
            tr_imgs = cfg["tfm"](X_cnn_train)
            vl_imgs = cfg["tfm"](X_cnn_val)
            print("Train images:", tr_imgs.shape, "Val images:", vl_imgs.shape)

            model, hist = train_cnn(
                cfg["model"],
                tr_imgs,
                vl_imgs,
                np.asarray(y_cnn_train),
                np.asarray(y_cnn_val),
                cfg["name"],
            )
            cnn_histories.append(hist)

            val_p = predict_cnn_in_chunks(model, cfg["tfm"], X_cnn_val)
            test_p = predict_cnn_in_chunks(model, cfg["tfm"], X_cnn_test)
            test_p_cal = platt_calibrate(y_cnn_val, val_p, test_p)

            val_preds[cfg["name"]] = val_p
            test_preds[cfg["name"]] = test_p_cal
            model_objects[cfg["name"]] = model

            scope = "cnn_test_subset" if CNN_TEST_MAX is not None else "full_test"
            results.append(evaluate_model(y_cnn_test, test_p_cal, cfg["name"], scope))

            safe_name = cfg["name"].lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            hist.to_csv(os.path.join(OUT_DIR, safe_name + "_history.csv"), index=False)
            torch.save(model.state_dict(), os.path.join(OUT_DIR, safe_name + ".pth"))

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(hist["epoch"], hist["loss"], label="Train focal loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(cfg["name"] + " - Training Loss")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, safe_name + "_loss.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(8, 5), dpi=200)
            plt.plot(hist["epoch"], hist["val_auc"], label="Validation AUROC")
            plt.xlabel("Epoch")
            plt.ylabel("AUROC")
            plt.title(cfg["name"] + " - Validation AUROC")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, safe_name + "_val_auc.png"), dpi=300)
            plt.close()

            del tr_imgs, vl_imgs
            free_memory()

        except Exception as e:
            import traceback
            print("[ERR] CNN branch failed:", cfg["name"], e)
            traceback.print_exc()
            free_memory()

# ============================================================
# 10. HYBRID STACKING
# ============================================================

print("\n[11/18] Hybrid stacking...")

# Full hybrid stack is only statistically clean when all predictions are on the same full-test set.
# If CNN_TEST_MAX is capped, we avoid mixing full ML predictions and subset CNN predictions in one reported model.
if RUN_CNN and CNN_TEST_MAX is None:
    candidate_models = [m for m in test_preds.keys() if m in val_preds]
    if len(candidate_models) >= 2:
        stack_val_all = np.column_stack([val_preds[m] for m in candidate_models])
        stack_test_all = np.column_stack([test_preds[m] for m in candidate_models])
        meta_all = LogisticRegression(C=0.3, max_iter=2000, random_state=SEED)
        meta_all.fit(stack_val_all, y_val)
        hybrid_test = meta_all.predict_proba(stack_test_all)[:, 1]
        test_preds["Hybrid ML+CNN Stack"] = hybrid_test
        results.append(evaluate_model(y_test, hybrid_test, "Hybrid ML+CNN Stack", "full_test"))
else:
    print("[INFO] Hybrid full-test stack skipped because CNN test is capped.")
    print("For full hybrid stack, set CNN_TEST_MAX=None, but runtime can be much longer.")

# ============================================================
# 11. SAVE METRICS
# ============================================================

print("\n[12/18] Saving metrics...")

results_df = pd.DataFrame(results).sort_values(["Eval_Scope", "AUROC"], ascending=[True, False])
metrics_path = os.path.join(OUT_DIR, "final_scopus_metrics.csv")
results_df.to_csv(metrics_path, index=False)
print("\nFINAL RESULTS")
print(results_df)
print("Saved:", metrics_path)

# ============================================================
# 12. ROC / PR / CALIBRATION PLOTS
# ============================================================

print("\n[13/18] Plotting ROC / PR / calibration...")

full_test_model_names = [m for m in ["XGBoost", "LightGBM", "CatBoost", "ML Stacked Ensemble", "Hybrid ML+CNN Stack"] if m in test_preds]
plot_preds_full = {m: test_preds[m] for m in full_test_model_names}

plt.figure(figsize=(8, 6), dpi=200)
for name, pred in plot_preds_full.items():
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    plt.plot(fpr, tpr, label=f"{name} AUROC={auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Full Test Models")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves_full_test.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6), dpi=200)
for name, pred in plot_preds_full.items():
    prec, rec, _ = precision_recall_curve(y_test, pred)
    auprc = average_precision_score(y_test, pred)
    plt.plot(rec, prec, label=f"{name} AUPRC={auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Full Test Models")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curves_full_test.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6), dpi=200)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
for name, pred in plot_preds_full.items():
    frac_pos, mean_pred = calibration_curve(y_test, pred, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_test, pred)
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} Brier={brier:.3f}")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed event fraction")
plt.title("Calibration Curves - Full Test Models")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calibration_full_test.png"), dpi=300)
plt.close()

# CNN subset plots.
if RUN_CNN and "y_cnn_test" in globals():
    cnn_names = [m for m in test_preds.keys() if m.startswith("CNN")]
    if cnn_names:
        plt.figure(figsize=(8, 6), dpi=200)
        for name in cnn_names:
            pred = test_preds[name]
            fpr, tpr, _ = roc_curve(y_cnn_test, pred)
            auc = roc_auc_score(y_cnn_test, pred)
            plt.plot(fpr, tpr, label=f"{name} AUROC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("CNN ROC Curves - CNN Test Subset")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cnn_roc_curves_test_subset.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6), dpi=200)
        for name in cnn_names:
            pred = test_preds[name]
            prec, rec, _ = precision_recall_curve(y_cnn_test, pred)
            auprc = average_precision_score(y_cnn_test, pred)
            plt.plot(rec, prec, label=f"{name} AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("CNN Precision-Recall Curves - CNN Test Subset")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "cnn_pr_curves_test_subset.png"), dpi=300)
        plt.close()

# ============================================================
# 13. FEATURE IMPORTANCE AND SHAP
# ============================================================

print("\n[14/18] Feature importance and SHAP...")

fi = pd.DataFrame({"feature": X_train.columns, "importance": xgb_model.feature_importances_}).sort_values("importance", ascending=False)
fi_path = os.path.join(OUT_DIR, "xgboost_feature_importance.csv")
fi.to_csv(fi_path, index=False)
print("\nTOP 30 FEATURES")
print(fi.head(30))

plt.figure(figsize=(8, 10), dpi=200)
plt.barh(fi.head(25)["feature"][::-1], fi.head(25)["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 25 XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xgboost_feature_importance.png"), dpi=300)
plt.close()

if RUN_SHAP and HAS_SHAP:
    try:
        print("Generating SHAP...")
        shap_sample = X_test.sample(min(SHAP_SAMPLE_N, len(X_test)), random_state=SEED)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(shap_sample)
        plt.figure(figsize=(10, 8), dpi=200)
        shap.summary_plot(shap_values, shap_sample, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "xgboost_shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[WARN] SHAP failed:", e)

# ============================================================
# 14. FAST MODEL COMPARISON - BOOTSTRAP INSTEAD OF FULL DELONG
# ============================================================

print("\n[15/18] Fast bootstrap model comparison on sampled full-test models...")

comparison_df = pd.DataFrame()

if RUN_FAST_MODEL_COMPARISON and len(plot_preds_full) >= 2:
    rng = np.random.RandomState(SEED)
    sample_n = min(MODEL_COMPARE_SAMPLE_N, len(y_test))
    idx = rng.choice(len(y_test), sample_n, replace=False)
    y_cmp = np.asarray(y_test)[idx]

    auc_map = {name: roc_auc_score(y_cmp, np.asarray(pred)[idx]) for name, pred in plot_preds_full.items()}
    best_name = max(auc_map, key=auc_map.get)
    best_pred = np.asarray(plot_preds_full[best_name])[idx]

    rows = []
    for name, pred in plot_preds_full.items():
        if name == best_name:
            continue
        other_pred = np.asarray(pred)[idx]
        auc_best = roc_auc_score(y_cmp, best_pred)
        auc_other = roc_auc_score(y_cmp, other_pred)
        diff = auc_best - auc_other

        boot_diffs = []
        for _ in range(MODEL_COMPARE_BOOTSTRAPS):
            bidx = rng.choice(len(y_cmp), len(y_cmp), replace=True)
            if len(np.unique(y_cmp[bidx])) < 2:
                continue
            boot_diffs.append(
                roc_auc_score(y_cmp[bidx], best_pred[bidx]) -
                roc_auc_score(y_cmp[bidx], other_pred[bidx])
            )
        boot_diffs = np.asarray(boot_diffs)
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_approx = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))

        rows.append({
            "Comparison": f"{best_name} vs {name}",
            "Sample_N": int(sample_n),
            "AUROC_Best": float(auc_best),
            "AUROC_Other": float(auc_other),
            "AUROC_Diff": float(diff),
            "Bootstrap_CI_Low": float(ci_low),
            "Bootstrap_CI_High": float(ci_high),
            "Approx_p_value": float(p_approx),
            "Significant_0.05": bool(p_approx < 0.05),
        })

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(OUT_DIR, "fast_model_comparison_bootstrap.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(comparison_df)
    print("Saved:", comparison_path)
else:
    print("[INFO] Fast model comparison skipped.")

# ============================================================
# 15. SUBGROUP ANALYSIS
# ============================================================

print("\n[16/18] Subgroup analysis...")

subgroup_df = pd.DataFrame()

if RUN_SUBGROUP:
    subgroup_rows = []
    Xt = X_test.reset_index(drop=True)
    yt_all = np.asarray(y_test)

    def add_subgroup(name, mask):
        mask = np.asarray(mask)
        if mask.sum() < 1000:
            return
        yt = yt_all[mask]
        if len(np.unique(yt)) < 2:
            return
        for model_name, pred in plot_preds_full.items():
            yp = np.asarray(pred)[mask]
            subgroup_rows.append({
                "Subgroup": name,
                "Model": model_name,
                "N": int(mask.sum()),
                "Preterm_Rate": float(yt.mean()),
                "AUROC": float(roc_auc_score(yt, yp)),
                "AUPRC": float(average_precision_score(yt, yp)),
                "Brier": float(brier_score_loss(yt, yp)),
            })

    if "plurality" in Xt.columns:
        add_subgroup("Singleton", Xt["plurality"].values == 1)
        add_subgroup("Multiple pregnancy", Xt["plurality"].values > 1)

    age_col = next((c for c in ["Mother_Age", "mother_age", "MAGER", "mager"] if c in Xt.columns), None)
    if age_col is not None:
        add_subgroup("Mother age < 20", Xt[age_col].values < 20)
        add_subgroup("Mother age 20-34", (Xt[age_col].values >= 20) & (Xt[age_col].values <= 34))
        add_subgroup("Mother age >= 35", Xt[age_col].values >= 35)

    smoke_col = next((c for c in ["Any_Smoking", "any_smoking", "smoking", "CIG_0"] if c in Xt.columns), None)
    if smoke_col is not None:
        add_subgroup("Non-smoking", Xt[smoke_col].values == 0)
        add_subgroup("Smoking", Xt[smoke_col].values > 0)

    htn_col = next((c for c in ["Gestational_Hypertension", "gestational_hypertension", "RF_GHYPE"] if c in Xt.columns), None)
    if htn_col is not None:
        add_subgroup("No gestational hypertension", Xt[htn_col].values == 0)
        add_subgroup("Gestational hypertension", Xt[htn_col].values > 0)

    bmi_col = next((c for c in ["Mother_BMI", "BMI"] if c in Xt.columns), None)
    if bmi_col is not None:
        add_subgroup("BMI < 18.5", Xt[bmi_col].values < 18.5)
        add_subgroup("BMI 18.5-29.9", (Xt[bmi_col].values >= 18.5) & (Xt[bmi_col].values < 30))
        add_subgroup("BMI >= 30", Xt[bmi_col].values >= 30)

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_path = os.path.join(OUT_DIR, "subgroup_analysis.csv")
    subgroup_df.to_csv(subgroup_path, index=False)
    print(subgroup_df.head(80))
    print("Saved:", subgroup_path)
else:
    print("[INFO] Subgroup analysis skipped.")

# ============================================================
# 16. DECISION CURVE ANALYSIS
# ============================================================

print("\n[17/18] Decision Curve Analysis...")

dca_df = pd.DataFrame()

if RUN_DCA:
    dca_rows = []
    thresholds = np.linspace(0.01, 0.50, 50)
    y_true_np = np.asarray(y_test)
    n = len(y_true_np)

    for model_name, pred in plot_preds_full.items():
        pred = np.asarray(pred)
        for pt in thresholds:
            y_pred = pred >= pt
            tp = ((y_pred == 1) & (y_true_np == 1)).sum()
            fp = ((y_pred == 1) & (y_true_np == 0)).sum()
            net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
            dca_rows.append({
                "Model": model_name,
                "Threshold": float(pt),
                "Net_Benefit": float(net_benefit),
            })

    dca_df = pd.DataFrame(dca_rows)
    dca_path = os.path.join(OUT_DIR, "decision_curve_analysis.csv")
    dca_df.to_csv(dca_path, index=False)

    plt.figure(figsize=(8, 6), dpi=200)
    for model_name in dca_df["Model"].unique():
        tmp = dca_df[dca_df["Model"] == model_name]
        plt.plot(tmp["Threshold"], tmp["Net_Benefit"], label=model_name)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_curve_analysis.png"), dpi=300)
    plt.close()

    print("Saved:", dca_path)
else:
    print("[INFO] DCA skipped.")

# ============================================================
# 17. REPORT GENERATION
# ============================================================

print("\n[18/18] Writing publication report...")

if cnn_histories:
    cnn_hist_df = pd.concat(cnn_histories, ignore_index=True)
    cnn_hist_path = os.path.join(OUT_DIR, "cnn_training_history.csv")
    cnn_hist_df.to_csv(cnn_hist_path, index=False)

    plt.figure(figsize=(9, 6), dpi=200)
    for m in cnn_hist_df["Model"].unique():
        tmp = cnn_hist_df[cnn_hist_df["Model"] == m]
        plt.plot(tmp["epoch"], tmp["val_auc"], label=m)
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUROC")
    plt.title("CNN Validation AUROC History")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cnn_validation_auc_history.png"), dpi=300)
    plt.close()

report_path = os.path.join(OUT_DIR, "paper_summary_report_scopus.md")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Scopus-Ready Experimental Report: Preterm Birth Prediction\n\n")
    f.write("## Study Design\n\n")
    f.write("This pipeline evaluates preterm birth prediction using a hybrid experimental design combining classical tabular machine learning and tabular-to-image feature-map convolutional neural networks. The design emphasizes large-scale validation, leakage prevention, calibration, subgroup analysis, decision-curve analysis, and interpretable feature importance.\n\n")

    f.write("## Data and Splitting\n\n")
    f.write(f"- Dataset source: `{DATASET_REF}`\n")
    f.write(f"- Training set: {X_train.shape[0]:,}\n")
    f.write(f"- Validation set: {X_val.shape[0]:,}\n")
    f.write(f"- Independent test set: {X_test.shape[0]:,}\n")
    f.write(f"- Training preterm rate: {float(y_train.mean()):.4f}\n")
    f.write(f"- Test preterm rate: {float(y_test.mean()):.4f}\n\n")

    f.write("## Leakage Prevention\n\n")
    f.write("The following target-derived, postnatal, delivery, or outcome-related variables were excluded prior to model training:\n\n")
    for c in LEAKAGE_COLS:
        f.write(f"- `{c}`\n")
    f.write("\n")

    f.write("## Models\n\n")
    f.write("### Full-test tabular ML models\n")
    f.write("- XGBoost\n- LightGBM\n- CatBoost\n- Logistic stacked ensemble\n\n")
    f.write("### Feature-map CNN models\n")
    f.write("- ResNet18-DeepInsight feature map\n")
    f.write("- EfficientNetB0-Fingerprint feature map\n")
    f.write("- SimpleCNN-UMAP highlight map\n\n")
    f.write("CNNs use an AMP-safe focal loss with logits; final probabilities are obtained via sigmoid at inference.\n\n")

    if RUN_CNN and cnn_context:
        f.write("## CNN Runtime Caps\n\n")
        f.write(f"- CNN training rows: {cnn_context.get('X_cnn_train_N', 'NA'):,}\n")
        f.write(f"- CNN validation rows: {cnn_context.get('X_cnn_val_N', 'NA'):,}\n")
        f.write(f"- CNN test rows: {cnn_context.get('X_cnn_test_N', 'NA'):,}\n")
        f.write(f"- CNN_TEST_MAX: {cnn_context.get('CNN_TEST_MAX')}\n\n")
        f.write("When CNN_TEST_MAX is capped, CNN results are reported as subset performance and are not directly stacked with full-test ML predictions.\n\n")

    f.write("## Final Model Performance\n\n")
    f.write(results_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Statistical Comparison\n\n")
    if not comparison_df.empty:
        f.write("Pairwise AUROC differences were estimated using bootstrap resampling on a stratified/random sampled subset of the independent test cohort to avoid computational infeasibility of full-test DeLong computation.\n\n")
        f.write(comparison_df.to_markdown(index=False))
    else:
        f.write("Statistical comparison was not generated.\n")
    f.write("\n\n")

    f.write("## Top XGBoost Features\n\n")
    f.write(fi.head(30).to_markdown(index=False))
    f.write("\n\n")

    if not subgroup_df.empty:
        f.write("## Subgroup Analysis Preview\n\n")
        f.write(subgroup_df.head(80).to_markdown(index=False))
        f.write("\n\n")

    if not dca_df.empty:
        f.write("## Decision Curve Analysis\n\n")
        f.write("Decision curve analysis was generated across threshold probabilities from 0.01 to 0.50. See `decision_curve_analysis.csv` and `decision_curve_analysis.png`.\n\n")

    f.write("## Recommended Manuscript Framing\n\n")
    f.write("If feature-map CNNs do not outperform tree-based ensembles, the manuscript should be framed as a large-scale benchmark and feasibility study of tabular-to-image representation learning for preterm birth prediction, rather than as a superiority claim for CNNs.\n")

print("Saved report:", report_path)

print("\n" + "=" * 90)
print("PIPELINE FINISHED")
print("All outputs saved to:", OUT_DIR)
print("Key files:")
print(" - final_scopus_metrics.csv")
print(" - paper_summary_report_scopus.md")
print(" - roc_curves_full_test.png")
print(" - pr_curves_full_test.png")
print(" - calibration_full_test.png")
print(" - xgboost_feature_importance.csv / .png")
print(" - xgboost_shap_summary.png")
print(" - fast_model_comparison_bootstrap.csv")
print(" - subgroup_analysis.csv")
print(" - decision_curve_analysis.csv / .png")
print(" - CNN checkpoints/history/plots if CNN is enabled")
print("=" * 90)
