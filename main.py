"""
main.py
=======
CASSAVA YAPRAK HASTALIĞI SINIFLANDIRMA — TAM PIPELINE

YAPI (3 NET BÖLÜM):
  KISIM A: Genel ayarlar (path, seed, device, EDA)
  KISIM B: Deep model ayarları (Optuna search space, 3-fold CV)
  KISIM C: Klasik ML ayarları (raw 64×64 grayscale → PCA → RF/ET/LR/SGD)

ÇIKTILAR (results/):
  - all_trials.csv          : Her trial bir satır (TÜM ayarlar + TÜM metrikler)
  - cm_*.png / cm_*.csv     : Confusion matrix (her trial)
  - perclass_*.png          : Sınıf bazlı precision/recall/f1 bar chart
  - curves_*.png            : Loss/accuracy training curves (deep için)
  - report_*.txt            : sklearn classification_report
  - 99_overall_comparison.png : Tüm trial'ların karşılaştırması

KULLANIM:
  python main.py             # Tüm pipeline (EDA + deep Optuna + klasik ML)
  python main.py --eda       # Sadece EDA
  python main.py --deep      # Sadece deep Optuna search
  python main.py --classical # Sadece klasik ML
  python main.py --dry-run   # Search space'i listele, çalıştırma
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

import joblib

from models import (
    build_model, fit_classical_model,
    build_loss, build_optimizer, build_scheduler, is_plateau_scheduler,
    build_eval_transform, build_classical_transform,
    AVAILABLE_MODELS, MODEL_TYPE, model_info,
    IMAGE_SIZE_CLASSICAL, IMAGE_SIZE_DEEP, IMAGE_SIZE_VIT,
)

# Optuna (sadece deep için)
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# KISIM A — GENEL AYARLAR
# =============================================================================

# ---- HANGİ MODEL TİPLERİ DENENECEK ----
# Seçenekler:
#   ["ML"]        → sadece klasik ML modelleri (CLASSICAL_MODELS listesi)
#   ["DL"]        → sadece deep learning modelleri (Optuna search)
#   ["ML", "DL"]  → önce ML, sonra DL (sırayla)
MODEL_TYPES = ["ML"]

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Path'ler ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CASSAVA_DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TRAIN_CSV = DATA_DIR / "train.csv"
LABEL_MAP_JSON = DATA_DIR / "label_num_to_disease_map.json"

# Çıktı dizinleri (Dataset/Results/ tüm CSV ve grafikleri içerir)
RESULTS_DIR = DATA_DIR / "Results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FEATURES_DIR = PROJECT_ROOT / "features"   # klasik ML için cache (raw piksel)
for d in (RESULTS_DIR, CHECKPOINTS_DIR, FEATURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

MASTER_CSV = RESULTS_DIR / "all_trials.csv"

# ---- Sınıf bilgileri ----
NUM_CLASSES = 5
CLASS_NAMES_FULL = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}
CLASS_NAMES_SHORT = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

# ---- Donanım ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 1)
USE_MIXED_PRECISION = True   # AMP — GPU'da float16 (~2x hız, ~%50 az bellek)

# ---- Cross Validation ----
N_FOLDS = 3   # 5-fold: Kaggle Cassava top çözümleri 5-fold kullandı; 3-fold daha gürültülü CV


# =============================================================================
# KISIM B — DEEP MODEL AYARLARI (Optuna ile hyperparameter tuning)
# =============================================================================
#
# Optuna nasıl çalışır?
#   - Her trial: search space'den categorical değerler seçer
#   - Her trial: 3-fold CV ile değerlendirilir
#   - TPE sampler: önceki trial'lardan öğrenir, daha iyi alanlara odaklanır
#   - MedianPruner: kötü trial'ları erken durdurur (zaman tasarrufu)
#
# UYARI: 50 trial × 5 fold = 250 deep run, her run ~30 dk → ~125 saat (5+ gün)
#        Pruning ile genelde ~%30-40 zaman tasarrufu olur.
#        Pratik öneri: OPTUNA_N_TRIALS=10-20 ile başlayın.

# ---- Deep eğitim sabitleri ----
DEEP_EPOCHS = 200
DEEP_EARLY_STOP_PATIENCE = 20
DEEP_GRAD_CLIP = 1.0   # gradient clipping: gürültülü etiket robustness için kritik

# ---- Optuna ayarları ----
OPTUNA_N_TRIALS = 50              # Toplam trial sayısı (10/20/50/100 önerilir)
OPTUNA_TIMEOUT_HOURS = None       # None = sınır yok; örn 24 = 24 saatte dur
OPTUNA_STUDY_NAME = "cassava_deep_search"
OPTUNA_OPTIMIZE_METRIC = "f1_macro_mean"   # ne maksimize edilecek

# Pruner: kötü performansı erken yakalama
OPTUNA_PRUNER_WARMUP_STEPS = 10    # ilk 10 epoch pruning kapalı
OPTUNA_PRUNER_N_STARTUP = 8       # ilk 8 trial'da pruning kapalı

# ---- DEEP SEARCH SPACE ----
# Optuna her trial'da bu listelerden categorical seçim yapacak
# Cassava Kaggle community + literatür-temelli en yaygın değerler

DEEP_SEARCH_SPACE = {
    # Hangi mimariler test edilecek
    "model": [
        "resnet50",
       # "efficientnet_b4",
       # "vit_b_16",
    ],

    # Learning rate (fine-tuning için)
    # 1e-4 standart, 5e-5 / 5e-4 sınır değerler
    "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],

    # Weight decay (L2 regularization)
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],

    # Batch size (GPU bellek + gradient stability)
    # 32 default; 16 küçük modeller için, 64 büyük GPU için
    "batch_size": [16, 32, 64],

    # ViT için ayrı batch_size havuzu (384×384 büyük → OOM riski)
    "batch_size_vit": [8, 16],

    # Augmentation — PHASE 1: Sadece "none" ile çalış.
    # En iyi hyperparameter kombinasyonu augmentation olmadan bulunur.
    # Faz 2 için augmentation_ablation.py kullanılır.
    "augmentation": ["none"],

    # Loss fonksiyonu — sınıf dengesizliği + GÜRÜLTÜLÜ ETİKET sorunu için
    # bi_tempered: Cassava'nın en etkili loss'u (1. ve 3. place kullandı)
    "loss": ["cross_entropy", "cross_entropy_ls", "focal", "bi_tempered"],

    # Loss-spesifik hyperparametreler (Optuna sadece ilgili olanları kullanır)
    "label_smoothing": [0.05, 0.10, 0.20],          # cross_entropy_ls için
    "focal_gamma":     [1.0, 1.5, 2.0, 2.5],         # focal için
    "bi_tempered_t1":  [0.6, 0.8, 1.0],              # bi_tempered için (0<t1<=1)
    "bi_tempered_t2":  [1.2, 1.4, 1.6],              # bi_tempered için (t2>=1)
    "bi_tempered_ls":  [0.0, 0.06, 0.1],             # bi_tempered label_smoothing

    # Optimizer
    "optimizer": ["adamw", "adam", "sgd"],
    "sgd_momentum": [0.9, 0.95, 0.99],   # sgd için

    # LR scheduler
    "scheduler": ["plateau", "cosine", "cosine_warm"],

    # Ablation: Sınıf ağırlıklı loss vs ağırlıksız loss
    # Project plan Section 2.5: "weighted vs unweighted loss"
    # True  → compute_class_weight('balanced') ile ağırlıklandır
    # False → eşit ağırlık (sınıf dengesizliğini görmezden gel)
    "use_class_weights": [True, False],
}


# =============================================================================
# KISIM C — KLASİK ML AYARLARI (raw piksel → PCA → RF/ET/LR/SGD)
# =============================================================================
#
# Klasik ML akışı:
#   1. Görüntüler → 64×64 grayscale → flatten → 4096-dim raw vektör
#   2. StandardScaler: sıfır-mean, birim-std normalizasyon
#   3. PCA(0.95): varyansın %95'ini koruyacak kadar bileşen
#                  (4096-dim → genellikle 150-300 dim arası iniyor)
#   4. Classifier: RandomForest / ExtraTrees / LogisticRegression / SGDClassifier
#
# NOT: Bu yaklaşımın deep modellerden çok daha düşük accuracy vereceği beklenir.
#      Yaprak hastalıkları yerel doku örüntüleriyle ayrılır; PCA global
#      varyansı tutar, texture'ı kaybeder. Karşılaştırma için iyi bir baseline.

# Hangi klasik modeller çalıştırılacak (her biri 3-fold CV ile)
CLASSICAL_MODELS = [
    "random_forest",
  #  "extra_trees",
   # "logistic_regression",
    "sgd_classifier",
]

# Raw piksel cache (bir kez çıkarılır, .npy olarak saklanır)
CLASSICAL_BATCH_SIZE = 256   # raw piksel yükleme — CPU için büyük olabilir

# Klasik ML hiperparametreleri models.py'de sabit (Optuna kullanılmıyor —
# çünkü zaten düşük performans bekliyoruz, hyperparameter tuning gereksiz)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed: int = RANDOM_SEED):
    """%100 tekrarlanır sonuçlar için tüm seed'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# DATASET
# =============================================================================

class CassavaDataset(Dataset):
    """Cassava görüntüleri için PyTorch Dataset."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["image_id"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Görüntü okunamadı: {img_path}\n{e}")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), row["image_id"]


def load_metadata():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"train.csv bulunamadı: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    if LABEL_MAP_JSON.exists():
        with open(LABEL_MAP_JSON) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}
    else:
        label_map = CLASS_NAMES_FULL
    return df, label_map


# =============================================================================
# EDA
# =============================================================================

def run_eda(df: pd.DataFrame, label_map: dict):
    print("\n" + "=" * 70)
    print("EDA — KEŞİFSEL VERİ ANALİZİ")
    print("=" * 70)
    print(f"Toplam görüntü:    {len(df)}")
    print(f"Sınıf sayısı:      {df['label'].nunique()}")
    print(f"Eksik veri:        {df.isnull().sum().sum()}")

    counts = df["label"].value_counts().sort_index()
    print(f"\n{'Lbl':<5}{'Sınıf':<42}{'Sayı':>8}{'Oran':>8}")
    print("-" * 63)
    for lbl, cnt in counts.items():
        name = label_map.get(lbl, CLASS_NAMES_FULL[lbl])
        print(f"{lbl:<5}{name:<42}{cnt:>8}{cnt/len(df)*100:>7.2f}%")
    print(f"\nDengesizlik (max/min): {counts.max()/counts.min():.2f}x")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#7F77DD", "#7F77DD", "#7F77DD", "#E24B4A", "#1D9E75"]
    bars = ax.bar(CLASS_NAMES_SHORT, counts.values, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Sınıf"); ax.set_ylabel("Görüntü sayısı")
    ax.set_title("Cassava Yaprak Hastalığı — Sınıf Dağılımı")
    ax.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{cnt}\n({cnt/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_class_distribution.png",
                dpi=150, bbox_inches="tight"); plt.close()

    # Sample images
    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(9, NUM_CLASSES * 2.5))
    for cls in range(NUM_CLASSES):
        cls_df = df[df["label"] == cls].sample(
            min(3, len(df[df["label"] == cls])), random_state=RANDOM_SEED)
        for j, (_, row) in enumerate(cls_df.iterrows()):
            try:
                img = Image.open(TRAIN_IMAGES_DIR / row["image_id"]).convert("RGB")
                axes[cls, j].imshow(img)
            except Exception:
                axes[cls, j].text(0.5, 0.5, "Yüklenemedi", ha="center", va="center")
            axes[cls, j].axis("off")
            if j == 0:
                axes[cls, j].set_ylabel(CLASS_NAMES_SHORT[cls], rotation=0,
                                        ha="right", va="center", fontweight="bold")
    plt.suptitle("Her Sınıftan Örnek Görüntüler", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_class_samples.png",
                dpi=120, bbox_inches="tight"); plt.close()
    print(f"✓ EDA çıktıları: {RESULTS_DIR}")


# =============================================================================
# METRİKLER
# =============================================================================

def compute_metrics(y_true, y_pred):
    """Accuracy + per-class + macro/weighted precision/recall/f1."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        average=None, zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "per_class_precision": p_per.tolist(),
        "per_class_recall": r_per.tolist(),
        "per_class_f1": f1_per.tolist(),
        "support": support.tolist(),
    }


def aggregate_fold_metrics(fold_metrics_list):
    """N fold'un metriklerini ortalama ± std olarak özetle."""
    out = {}
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
              "precision_weighted", "recall_weighted", "f1_weighted"]:
        vals = [m[k] for m in fold_metrics_list]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    for k in ["per_class_precision", "per_class_recall", "per_class_f1"]:
        vals = np.array([m[k] for m in fold_metrics_list])
        out[f"{k}_mean"] = vals.mean(axis=0).tolist()
        out[f"{k}_std"] = vals.std(axis=0).tolist()
    return out


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[0], cbar=False)
    axes[0].set_title(f"{title}\nConfusion Matrix (Sayılar)")
    axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gerçek")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[1])
    axes[1].set_title(f"{title}\nNormalize"); axes[1].set_xlabel("Tahmin")
    axes[1].set_ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_csv(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    df = pd.DataFrame(cm, index=CLASS_NAMES_SHORT,
                      columns=[f"pred_{c}" for c in CLASS_NAMES_SHORT])
    df.index.name = "true_class"
    df.to_csv(RESULTS_DIR / filename)


def plot_per_class_metrics(val_metrics: dict, train_metrics: dict,
                           title: str, filename: str):
    """Sınıf bazlı precision/recall/F1: TRAIN ve VAL yan yana 2 panelde."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    x = np.arange(NUM_CLASSES); w = 0.25

    for ax, m, label in zip(axes,
                            [train_metrics, val_metrics],
                            ["TRAIN", "VALIDATION"]):
        if not m:
            ax.set_visible(False)
            continue
        p = m["per_class_precision_mean"]
        r = m["per_class_recall_mean"]
        f1 = m["per_class_f1_mean"]
        ax.bar(x - w, p, w, label="Precision", color="#7F77DD")
        ax.bar(x,     r, w, label="Recall",    color="#1D9E75")
        ax.bar(x + w, f1, w, label="F1",        color="#E24B4A")
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES_SHORT)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Skor")
        ax.set_title(f"{label}\nSınıf Bazlı Performans ({N_FOLDS}-fold ortalama)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(p):  ax.text(i - w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for i, v in enumerate(r):  ax.text(i,     v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for i, v in enumerate(f1): ax.text(i + w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight"); plt.close()


def plot_training_curves(histories: list, title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fold_idx, h in enumerate(histories, start=1):
        epochs = list(range(1, len(h["train_loss"]) + 1))
        axes[0].plot(epochs, h["train_loss"], "--", alpha=0.6,
                     label=f"Fold {fold_idx} train")
        axes[0].plot(epochs, h["val_loss"], "-", label=f"Fold {fold_idx} val")
        axes[1].plot(epochs, h["train_acc"], "--", alpha=0.6,
                     label=f"Fold {fold_idx} train")
        axes[1].plot(epochs, h["val_acc"], "-", label=f"Fold {fold_idx} val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight"); plt.close()


def plot_overall_comparison(results_df: pd.DataFrame):
    """Tüm trial'lar için TRAIN ve VAL metriklerini yan yana karşılaştır."""
    if len(results_df) == 0:
        return
    rdf = results_df.copy()
    trial_col = "trial_no" if "trial_no" in rdf.columns else "trial_id"
    rdf["combo"] = rdf[trial_col].astype(str) + "\n" + rdf["model"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(max(15, len(rdf) * 0.6), 10))
    axes = axes.flatten()
    base_keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    lbls = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
    n = len(rdf); xpos = np.arange(n); w = 0.4
    for ax, base, ml in zip(axes, base_keys, lbls):
        train_mean = f"train_{base}_mean"
        val_mean = f"val_{base}_mean"
        train_std = f"train_{base}_std"
        val_std = f"val_{base}_std"
        if train_mean not in rdf.columns and val_mean not in rdf.columns:
            ax.set_visible(False); continue
        if train_mean in rdf.columns:
            ax.bar(xpos - w/2, rdf[train_mean].fillna(0).values, w,
                   yerr=rdf.get(train_std, pd.Series([0]*n)).fillna(0).values,
                   capsize=3, label="Train", color="#7F77DD",
                   edgecolor="black", linewidth=0.5)
        if val_mean in rdf.columns:
            ax.bar(xpos + w/2, rdf[val_mean].fillna(0).values, w,
                   yerr=rdf.get(val_std, pd.Series([0]*n)).fillna(0).values,
                   capsize=3, label="Validation", color="#E24B4A",
                   edgecolor="black", linewidth=0.5)
        ax.set_title(ml); ax.set_ylabel(ml); ax.set_ylim(0, 1.05)
        ax.set_xticks(xpos); ax.set_xticklabels(rdf["combo"].values, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle("Tüm Trial'ların Karşılaştırması (Train vs Validation)",
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "99_overall_comparison.png",
                dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
# CSV ROW BUILDER — Tüm ayarları + tüm metrikleri tek satıra yaz
# =============================================================================

# Sabit metrik anahtarları (sıra önemli — CSV sütun sırasını belirler)
_AGG_METRIC_KEYS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
]


def _flatten_metrics(summary: dict, prefix: str) -> dict:
    """
    Bir aggregate summary dict'ini düz CSV sütunlarına çevirir.
    prefix: 'train' veya 'val' (her sütun bu önekle başlar).
    """
    out = {}
    if not summary:
        return out
    # Genel metrikler
    for k in _AGG_METRIC_KEYS:
        if f"{k}_mean" in summary:
            out[f"{prefix}_{k}_mean"] = round(summary[f"{k}_mean"], 4)
        if f"{k}_std" in summary:
            out[f"{prefix}_{k}_std"] = round(summary[f"{k}_std"], 4)
    # Per-class metrikler
    for cls_idx, cls_name in enumerate(CLASS_NAMES_SHORT):
        for metric in ["precision", "recall", "f1"]:
            key_mean = f"per_class_{metric}_mean"
            key_std = f"per_class_{metric}_std"
            if key_mean in summary:
                out[f"{prefix}_{metric}_{cls_name}_mean"] = round(
                    summary[key_mean][cls_idx], 4)
            if key_std in summary:
                out[f"{prefix}_{metric}_{cls_name}_std"] = round(
                    summary[key_std][cls_idx], 4)
    return out


def build_csv_row(trial_id: str, trial_params: dict,
                  val_summary: dict, train_summary: dict,
                  best_epochs: list,
                  wall_time_sec: float, model_type: str,
                  optuna_trial_number: int = -1,
                  status: str = "completed",
                  pruned: bool = False) -> dict:
    """
    Her trial için CSV satırı oluştur. Sütun SIRASI:
      1) trial_no, model, augment, size
      2) Tüm parametreler ve hyperparametreler
      3) train metrikleri (train_*)
      4) validation metrikleri (val_*)
    """
    image_size = (IMAGE_SIZE_DEEP if model_type == "deep"
                  else IMAGE_SIZE_CLASSICAL)

    # 1) Kimlik + temel sütunlar (kullanıcı isteği: trial_no, model, augment, size)
    row = {
        "trial_no": trial_id,
        "model": trial_params.get("model", "-"),
        "augment": trial_params.get("augmentation", "-"),
        "size": image_size,
    }

    # 2) Tüm parametreler ve hyperparametreler
    row.update({
        "model_type": model_type,
        "optuna_trial_number": optuna_trial_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "optuna_pruned": pruned,
        "n_classes": NUM_CLASSES,
        "n_folds": N_FOLDS,
        "input_format": ("RGB ImageNet-normalized" if model_type == "deep"
                         else "64x64 grayscale flatten -> StandardScaler -> PCA(0.95)"),
        # Hiperparametreler
        "lr": trial_params.get("lr", "-"),
        "weight_decay": trial_params.get("weight_decay", "-"),
        "batch_size": trial_params.get("batch_size", "-"),
        "epochs_max": DEEP_EPOCHS if model_type == "deep" else "-",
        "early_stop_patience": (DEEP_EARLY_STOP_PATIENCE
                                if model_type == "deep" else "-"),
        "loss_name": trial_params.get("loss_name", "-"),
        "loss_kwargs": json.dumps(trial_params.get("loss_kwargs", {})),
        "optimizer_name": trial_params.get("optimizer_name", "-"),
        "optimizer_kwargs": json.dumps(trial_params.get("optimizer_kwargs", {})),
        "scheduler_name": trial_params.get("scheduler_name", "-"),
        "scheduler_kwargs": json.dumps(trial_params.get("scheduler_kwargs", {})),
        "seed": RANDOM_SEED,
        "device": str(DEVICE),
        "mixed_precision": USE_MIXED_PRECISION and DEVICE.type == "cuda",
        "use_class_weights": trial_params.get("use_class_weights", True),
        "class_weight": "balanced" if trial_params.get("use_class_weights", True)
                        else "none",
        "best_epoch_per_fold": json.dumps([e for e in best_epochs if e is not None]),
        "wall_time_min": round(wall_time_sec / 60, 2),
    })

    # 3) Train metrikleri
    row.update(_flatten_metrics(train_summary, prefix="train"))
    # 4) Validation metrikleri
    row.update(_flatten_metrics(val_summary, prefix="val"))
    return row


def append_to_master_csv(row: dict):
    """Trial sonucunu master CSV'ye incremental ekle. Sütun sırası KORUNUR.

    Yeni sütun sırası: row dict'inin insertion order'ı (Python 3.7+).
    Mevcut CSV'de olup yeni satırda olmayan sütunlar sona eklenir.
    """
    df_new = pd.DataFrame([row])
    new_cols = list(row.keys())
    if MASTER_CSV.exists():
        df_existing = pd.read_csv(MASTER_CSV)
        # Yeni sütun sırası: önce row order, sonra existing-only kalanlar
        extra_cols = [c for c in df_existing.columns if c not in row]
        ordered_cols = new_cols + extra_cols
        df_existing = df_existing.reindex(columns=ordered_cols)
        df_new = df_new.reindex(columns=ordered_cols)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new[new_cols]
    df_combined.to_csv(MASTER_CSV, index=False)


# =============================================================================
# DEEP: TEK FOLD EĞİTİMİ
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for images, labels, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images); loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEEP_GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(images); loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEEP_GRAD_CLIP)
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate_deep(model, loader, criterion, use_amp):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images); loss = criterion(logits, labels)
        else:
            logits = model(images); loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total_n += images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (total_loss/total_n, total_correct/total_n,
            np.concatenate(all_preds), np.concatenate(all_labels))


@torch.no_grad()
def evaluate_deep_tta(model, dataset_df: "pd.DataFrame", images_dir,
                      batch_size: int, image_size: int, use_amp: bool):
    """
    Test Time Augmentation (TTA) ile inference.

    Uygulanan TTA dönüşümleri (1. place'den esinlenildi):
      - Orijinal görüntü (no aug)
      - Yatay flip
      - Dikey flip
    3 pasın softmax olasılıkları ortalaması alınır.

    Kaggle Cassava'da basit flip TTA ~+0.3-0.5% accuracy verir.
    """
    model.eval()
    tta_transforms = [
        build_eval_transform(image_size=image_size),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    ]

    all_probs = None
    all_labels = None
    for t_idx, transform in enumerate(tta_transforms):
        ds = CassavaDataset(dataset_df, images_dir, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=(DEVICE.type == "cuda"))
        probs_list, labels_list = [], []
        for images, labels, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
        probs_arr = np.concatenate(probs_list, axis=0)
        if all_probs is None:
            all_probs = probs_arr
            all_labels = np.concatenate(labels_list, axis=0)
        else:
            all_probs += probs_arr

    tta_preds = all_probs.argmax(axis=1)
    return tta_preds, all_labels


def train_deep_one_fold(params: dict, train_df, val_df, fold_idx: int):
    """Tek bir trial'ın tek fold'u."""
    set_all_seeds(RANDOM_SEED + fold_idx)

    train_ds = CassavaDataset(train_df, TRAIN_IMAGES_DIR,
                              transform=build_eval_transform(
                                  image_size=(IMAGE_SIZE_VIT
                                              if params["model"] == "vit_b_16"
                                              else IMAGE_SIZE_DEEP)))
    val_ds = CassavaDataset(val_df, TRAIN_IMAGES_DIR,
                            transform=build_eval_transform(
                                image_size=(IMAGE_SIZE_VIT
                                            if params["model"] == "vit_b_16"
                                            else IMAGE_SIZE_DEEP)))

    g = torch.Generator(); g.manual_seed(RANDOM_SEED + fold_idx)
    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"] * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Model
    model = build_model(params["model"], num_classes=NUM_CLASSES,
                        pretrained=True).to(DEVICE)

    # Class-balanced weights (ablation: use_class_weights=False → eşit ağırlık)
    if params.get("use_class_weights", True):
        cw = compute_class_weight(class_weight="balanced",
                                  classes=np.arange(NUM_CLASSES),
                                  y=train_df["label"].values)
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    else:
        cw_tensor = None  # unweighted: tüm sınıflar eşit ağırlıklı

    # Loss / Optimizer / Scheduler
    criterion = build_loss(params["loss_name"], class_weights=cw_tensor,
                           **params["loss_kwargs"])
    optimizer = build_optimizer(params["optimizer_name"], model.parameters(),
                                lr=params["lr"], weight_decay=params["weight_decay"],
                                **params["optimizer_kwargs"])
    scheduler = build_scheduler(params["scheduler_name"], optimizer,
                                total_epochs=DEEP_EPOCHS,
                                **params["scheduler_kwargs"])

    use_amp = USE_MIXED_PRECISION and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf"); best_state = None
    best_epoch = 0; patience_counter = 0

    print(f"\n  >> {params['model']} | Fold {fold_idx+1}/{N_FOLDS}")
    print(f"     lr={params['lr']:.0e} wd={params['weight_decay']:.0e} "
          f"bs={params['batch_size']} aug={params['augmentation']} "
          f"loss={params['loss_name']} opt={params['optimizer_name']} "
          f"sch={params['scheduler_name']} "
          f"cw={'Y' if params.get('use_class_weights', True) else 'N'}")

    for epoch in range(1, DEEP_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, use_amp)
        val_loss, val_acc, _, _ = evaluate_deep(model, val_loader, criterion, use_amp)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if scheduler is not None:
            if is_plateau_scheduler(scheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        improved = val_loss < best_val_loss
        marker = "★" if improved else " "
        print(f"     Ep {epoch:2d} {marker} "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={current_lr:.1e} | {time.time()-t0:.1f}s")

        if improved:
            best_val_loss = val_loss; best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEEP_EARLY_STOP_PATIENCE:
                print(f"     ⚠ Early stop @ ep{epoch} (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final değerlendirme: hem TRAIN hem VAL üzerinde (eval transform ile) ----
    # Train metriklerini augmentation'sız (eval transform) hesapla — overfit teşhisi için
    train_eval_ds = CassavaDataset(train_df, TRAIN_IMAGES_DIR,
                                   transform=build_eval_transform(
                                       image_size=(IMAGE_SIZE_VIT
                                                   if params["model"] == "vit_b_16"
                                                   else IMAGE_SIZE_DEEP)))
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=params["batch_size"] * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
    )
    _, _, train_preds, train_true = evaluate_deep(model, train_eval_loader,
                                                  criterion, use_amp)
    train_metrics = compute_metrics(train_true, train_preds)
    train_metrics["best_epoch"] = best_epoch

    _, _, val_preds, val_true = evaluate_deep(model, val_loader, criterion, use_amp)
    val_metrics = compute_metrics(val_true, val_preds)
    val_metrics["best_epoch"] = best_epoch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_metrics, train_metrics, history, val_preds, val_true, train_preds, train_true


# =============================================================================
# DEEP: 3-FOLD CV (BİR TRIAL TAM)
# =============================================================================

def run_deep_trial_3fold(params: dict, df: pd.DataFrame,
                         trial_id: str):
    """
    Bir deep trial — 3-fold CV. Tüm fold'ların ağırlıklı ortalamasını döndürür.

    Returns
    -------
    val_summary, train_summary : dict   # 3-fold aggregate metrikler
    best_epochs : list[int]             # her fold'un best epoch'u
    oof_true, oof_pred                  # tüm OOF predictions (val)
    train_true_all, train_pred_all      # tüm fold'ların train predictions (concatenated)
    histories : list[dict]              # her fold'un training history'si
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_all = df["label"].values
    indices = np.arange(len(df))

    val_fold_metrics, train_fold_metrics, histories = [], [], []
    oof_preds = np.zeros(len(df), dtype=np.int64)
    oof_true = np.zeros(len(df), dtype=np.int64)
    train_true_all, train_pred_all = [], []
    best_epochs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(indices, y_all)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        v_m, t_m, h, v_preds, v_true, t_preds, t_true = train_deep_one_fold(
            params, train_df, val_df, fold_idx)
        val_fold_metrics.append(v_m); train_fold_metrics.append(t_m)
        histories.append(h)
        oof_preds[va_idx] = v_preds; oof_true[va_idx] = v_true
        train_true_all.append(t_true); train_pred_all.append(t_preds)
        best_epochs.append(v_m["best_epoch"])
        print(f"     ⇒ Fold {fold_idx+1}: "
              f"train_acc={t_m['accuracy']:.4f} f1={t_m['f1_macro']:.4f} | "
              f"val_acc={v_m['accuracy']:.4f} f1={v_m['f1_macro']:.4f}")

    val_summary = aggregate_fold_metrics(val_fold_metrics)
    train_summary = aggregate_fold_metrics(train_fold_metrics)
    train_true_all = np.concatenate(train_true_all)
    train_pred_all = np.concatenate(train_pred_all)
    return (val_summary, train_summary, best_epochs,
            oof_true, oof_preds, train_true_all, train_pred_all, histories)


# =============================================================================
# OPTUNA OBJECTIVE (DEEP MODELLER İÇİN)
# =============================================================================

def optuna_objective(trial: "optuna.Trial", df: pd.DataFrame) -> float:
    """
    Optuna her trial'da bu fonksiyonu çağırır.
    Her trial:
      1. Search space'den categorical değer seçer
      2. 3-fold CV ile değerlendirir
      3. f1_macro_mean değerini maksimize etmeye çalışır
      4. CSV'ye satır ekler
    """
    t_trial_start = time.time()

    # ---- Search space'den seçimler ----
    model_name = trial.suggest_categorical("model", DEEP_SEARCH_SPACE["model"])
    lr = trial.suggest_categorical("lr", DEEP_SEARCH_SPACE["lr"])
    wd = trial.suggest_categorical("weight_decay", DEEP_SEARCH_SPACE["weight_decay"])
    bs = trial.suggest_categorical(
        "batch_size",
        DEEP_SEARCH_SPACE["batch_size_vit"] if model_name == "vit_b_16"
        else DEEP_SEARCH_SPACE["batch_size"])
    aug = trial.suggest_categorical("augmentation", DEEP_SEARCH_SPACE["augmentation"])
    loss_name = trial.suggest_categorical("loss", DEEP_SEARCH_SPACE["loss"])
    opt_name = trial.suggest_categorical("optimizer", DEEP_SEARCH_SPACE["optimizer"])
    sched_name = trial.suggest_categorical("scheduler", DEEP_SEARCH_SPACE["scheduler"])
    use_cw = trial.suggest_categorical("use_class_weights",
                                       DEEP_SEARCH_SPACE["use_class_weights"])

    # ---- Loss-spesifik params ----
    loss_kwargs = {}
    if loss_name == "cross_entropy_ls":
        loss_kwargs["label_smoothing"] = trial.suggest_categorical(
            "label_smoothing", DEEP_SEARCH_SPACE["label_smoothing"])
    elif loss_name == "focal":
        loss_kwargs["gamma"] = trial.suggest_categorical(
            "focal_gamma", DEEP_SEARCH_SPACE["focal_gamma"])
    elif loss_name == "bi_tempered":
        loss_kwargs["t1"] = trial.suggest_categorical(
            "bi_tempered_t1", DEEP_SEARCH_SPACE["bi_tempered_t1"])
        loss_kwargs["t2"] = trial.suggest_categorical(
            "bi_tempered_t2", DEEP_SEARCH_SPACE["bi_tempered_t2"])
        loss_kwargs["label_smoothing"] = trial.suggest_categorical(
            "bi_tempered_ls", DEEP_SEARCH_SPACE["bi_tempered_ls"])

    # ---- Optimizer-spesifik params ----
    optimizer_kwargs = {}
    if opt_name == "sgd":
        optimizer_kwargs["momentum"] = trial.suggest_categorical(
            "sgd_momentum", DEEP_SEARCH_SPACE["sgd_momentum"])
        optimizer_kwargs["nesterov"] = True

    # ---- Scheduler-spesifik params (sabit, search yok) ----
    scheduler_kwargs = {}
    if sched_name == "plateau":
        scheduler_kwargs = {"factor": 0.5, "patience": 2, "min_lr": 1e-7}
    elif sched_name == "cosine":
        scheduler_kwargs = {"eta_min": 1e-7}
    elif sched_name == "cosine_warm":
        scheduler_kwargs = {"T_0": 5, "T_mult": 2, "eta_min": 1e-7}

    # ---- Trial parametreleri toparla ----
    params = {
        "model": model_name,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": bs,
        "augmentation": aug,
        "loss_name": loss_name,
        "loss_kwargs": loss_kwargs,
        "optimizer_name": opt_name,
        "optimizer_kwargs": optimizer_kwargs,
        "scheduler_name": sched_name,
        "scheduler_kwargs": scheduler_kwargs,
        "use_class_weights": use_cw,
    }

    trial_id = f"optuna_{trial.number:03d}"
    print(f"\n{'#'*70}")
    print(f"# OPTUNA TRIAL #{trial.number} | {trial_id}")
    print(f"# Params: {params}")
    print(f"{'#'*70}")

    try:
        # ---- 3-fold CV ----
        (val_summary, train_summary, best_epochs,
         oof_true, oof_preds, train_true_all, train_pred_all,
         histories) = run_deep_trial_3fold(params, df, trial_id)

        # ---- Görseller ----
        title = f"{trial_id} | {model_name}"
        fname_base = f"{trial_id}_{model_name}"
        plot_confusion_matrix(oof_true, oof_preds, title, f"cm_{fname_base}.png")
        save_confusion_matrix_csv(oof_true, oof_preds, f"cm_{fname_base}.csv")
        plot_per_class_metrics(val_summary, train_summary, title,
                               f"perclass_{fname_base}.png")
        plot_training_curves(histories, title, f"curves_{fname_base}.png")

        report_val = classification_report(oof_true, oof_preds,
                                           target_names=CLASS_NAMES_SHORT,
                                           digits=4, zero_division=0)
        report_train = classification_report(train_true_all, train_pred_all,
                                             target_names=CLASS_NAMES_SHORT,
                                             digits=4, zero_division=0)
        with open(RESULTS_DIR / f"report_{fname_base}.txt", "w") as f:
            f.write(f"=== {title} ===\n\nParams: {params}\n\n"
                    f"--- TRAIN ---\n{report_train}\n\n"
                    f"--- VALIDATION ---\n{report_val}\n")

        # ---- CSV'ye yaz ----
        row = build_csv_row(trial_id, params, val_summary, train_summary,
                            best_epochs,
                            time.time() - t_trial_start, "deep",
                            optuna_trial_number=trial.number,
                            status="completed")
        append_to_master_csv(row)

        # ---- Optuna'ya hedef metriği döndür ----
        score = val_summary[OPTUNA_OPTIMIZE_METRIC]
        print(f"\n✓ Trial {trial.number} bitti: {OPTUNA_OPTIMIZE_METRIC}={score:.4f}")
        return score

    except Exception as e:
        print(f"\n✗ Trial {trial.number} hata verdi: {e}")
        import traceback; traceback.print_exc()
        # CSV'ye hatalı trial'ı da yaz
        row = build_csv_row(trial_id, params, {}, {}, [],
                            time.time() - t_trial_start, "deep",
                            optuna_trial_number=trial.number,
                            status=f"failed: {str(e)[:100]}")
        append_to_master_csv(row)
        # Optuna'nın bu trial'ı kötü olarak işaretlemesi için düşük skor döndür
        return 0.0


def run_optuna_study(df: pd.DataFrame, dry_run: bool = False):
    """Optuna study başlat."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna yüklü değil. pip install optuna")

    print("\n" + "=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH (DEEP MODELLER)")
    print("=" * 70)
    print(f"  Trial sayısı:        {OPTUNA_N_TRIALS}")
    print(f"  Her trial:           {N_FOLDS}-fold CV ({DEEP_EPOCHS} epoch + early stop)")
    print(f"  Optimize metrik:     {OPTUNA_OPTIMIZE_METRIC} (maximize)")
    print(f"  Sampler:             TPE (Tree-structured Parzen Estimator)")
    print(f"  Pruner:              MedianPruner (warmup={OPTUNA_PRUNER_WARMUP_STEPS})")
    print(f"  Search space:")
    for k, v in DEEP_SEARCH_SPACE.items():
        print(f"    {k:18s}: {v}")
    if OPTUNA_TIMEOUT_HOURS:
        print(f"  Zaman limiti:        {OPTUNA_TIMEOUT_HOURS} saat")

    if dry_run:
        print("\n--dry-run: Optuna study başlatılmadı.")
        return

    # Optuna study oluştur
    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(
        n_startup_trials=OPTUNA_PRUNER_N_STARTUP,
        n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_STUDY_NAME,
        sampler=sampler,
        pruner=pruner,
    )

    timeout_sec = OPTUNA_TIMEOUT_HOURS * 3600 if OPTUNA_TIMEOUT_HOURS else None
    study.optimize(
        lambda trial: optuna_objective(trial, df),
        n_trials=OPTUNA_N_TRIALS,
        timeout=timeout_sec,
        show_progress_bar=False,
    )

    # ---- Optuna sonuçlarını özetle ----
    print("\n" + "=" * 70)
    print("OPTUNA STUDY SONUÇLARI")
    print("=" * 70)
    print(f"Toplam trial: {len(study.trials)}")
    print(f"En iyi trial: #{study.best_trial.number}")
    print(f"En iyi skor:  {study.best_value:.4f} ({OPTUNA_OPTIMIZE_METRIC})")
    print(f"En iyi params:")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    # Optuna study'i kaydet (sonradan yüklenebilir)
    joblib.dump(study, CHECKPOINTS_DIR / "optuna_study.pkl")
    print(f"\n✓ Optuna study kaydedildi: {CHECKPOINTS_DIR / 'optuna_study.pkl'}")


# =============================================================================
# KLASİK ML: RAW PİKSEL ÇIKARMA + 3-FOLD CV
# =============================================================================

def extract_raw_pixels(df: pd.DataFrame, force: bool = False):
    """
    Tüm görüntülerden 64×64 grayscale flatten vektörler çıkar.
    .npy olarak cache eder.
    """
    cache_X = FEATURES_DIR / "raw_pixels_X.npy"
    cache_y = FEATURES_DIR / "raw_pixels_y.npy"

    if (not force) and cache_X.exists() and cache_y.exists():
        print(f"\n[Raw pixels] Cache'den yükleniyor...")
        return np.load(cache_X), np.load(cache_y)

    print(f"\n{'='*70}")
    print(f"RAW PİKSEL ÇIKARMA: {IMAGE_SIZE_CLASSICAL}×{IMAGE_SIZE_CLASSICAL} grayscale flatten")
    print(f"  Görüntü: {len(df)} | Çıktı boyutu: {IMAGE_SIZE_CLASSICAL**2}-dim")
    print(f"{'='*70}")

    transform = build_classical_transform(IMAGE_SIZE_CLASSICAL)
    dataset = CassavaDataset(df, TRAIN_IMAGES_DIR, transform=transform)
    loader = DataLoader(
        dataset, batch_size=CLASSICAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_X, all_y = [], []
    start = time.time()
    for images, labels, _ in tqdm(loader, desc="[raw pixels]"):
        # images: (B, 4096) tensörü
        all_X.append(images.numpy())
        all_y.append(labels.numpy())
    elapsed = time.time() - start

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    print(f"  Süre: {elapsed/60:.1f} dk | Shape: {X.shape} | "
          f"Bellek: {X.nbytes/1024**2:.1f} MB")

    np.save(cache_X, X)
    np.save(cache_y, y)
    print(f"✓ Cache'lendi: {FEATURES_DIR}")
    return X, y


def run_classical_trial(model_name: str, X: np.ndarray, y: np.ndarray,
                        trial_id: str) -> dict:
    """Klasik ML modeli için 3-fold CV."""
    print(f"\n{'='*70}")
    print(f"CLASSICAL TRIAL: {trial_id} | {model_name}")
    print(f"  Pipeline: StandardScaler → PCA(0.95) → {model_name}")
    print(f"  Input shape: {X.shape}")
    print(f"{'='*70}")
    t_trial_start = time.time()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    val_fold_metrics, train_fold_metrics = [], []
    oof_preds = np.zeros_like(y); oof_true = np.zeros_like(y)
    train_true_all, train_pred_all = [], []
    fit_times = []
    pca_components = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = build_model(model_name)  # her fold YENİ pipeline
        t0 = time.time()
        model = fit_classical_model(model_name, model, X_tr, y_tr)
        fit_times.append(time.time() - t0)

        # PCA bileşen sayısını logla
        n_pca = model.named_steps["pca"].n_components_
        pca_components.append(n_pca)

        val_preds = model.predict(X_va)
        train_preds = model.predict(X_tr)
        v_m = compute_metrics(y_va, val_preds)
        t_m = compute_metrics(y_tr, train_preds)
        val_fold_metrics.append(v_m); train_fold_metrics.append(t_m)
        oof_preds[va_idx] = val_preds; oof_true[va_idx] = y_va
        train_true_all.append(y_tr); train_pred_all.append(train_preds)
        print(f"  Fold {fold_idx}: "
              f"train_acc={t_m['accuracy']:.4f} f1={t_m['f1_macro']:.4f} | "
              f"val_acc={v_m['accuracy']:.4f} f1={v_m['f1_macro']:.4f} | "
              f"PCA bileşen: {n_pca} | fit: {fit_times[-1]:.1f}s")

        if fold_idx == N_FOLDS:
            joblib.dump(model, CHECKPOINTS_DIR /
                        f"{trial_id}_{model_name}_lastfold.pkl")

    val_summary = aggregate_fold_metrics(val_fold_metrics)
    train_summary = aggregate_fold_metrics(train_fold_metrics)
    train_true_all = np.concatenate(train_true_all)
    train_pred_all = np.concatenate(train_pred_all)

    # Görseller
    title = f"{trial_id} | {model_name}"
    fname_base = f"{trial_id}_{model_name}"
    plot_confusion_matrix(oof_true, oof_preds, title, f"cm_{fname_base}.png")
    save_confusion_matrix_csv(oof_true, oof_preds, f"cm_{fname_base}.csv")
    plot_per_class_metrics(val_summary, train_summary, title,
                           f"perclass_{fname_base}.png")

    report_val = classification_report(oof_true, oof_preds,
                                       target_names=CLASS_NAMES_SHORT,
                                       digits=4, zero_division=0)
    report_train = classification_report(train_true_all, train_pred_all,
                                         target_names=CLASS_NAMES_SHORT,
                                         digits=4, zero_division=0)
    with open(RESULTS_DIR / f"report_{fname_base}.txt", "w") as f:
        f.write(f"=== {title} ===\n\n"
                f"--- TRAIN ---\n{report_train}\n\n"
                f"--- VALIDATION ---\n{report_val}\n")

    # CSV'ye yaz
    params = {
        "model": model_name,
        "lr": "-", "weight_decay": "-", "batch_size": "-",
        "augmentation": "-",
        "loss_name": "class_balanced (model-internal)",
        "loss_kwargs": {},
        "optimizer_name": "-", "optimizer_kwargs": {},
        "scheduler_name": "-", "scheduler_kwargs": {},
    }
    row = build_csv_row(trial_id, params, val_summary, train_summary,
                        [None]*N_FOLDS,
                        time.time() - t_trial_start, "classical")
    row["fit_time_mean_sec"] = round(float(np.mean(fit_times)), 2)
    row["pca_components_mean"] = round(float(np.mean(pca_components)), 1)
    append_to_master_csv(row)
    return row


def run_classical_pipeline(df: pd.DataFrame, force: bool = False):
    """Tüm klasik ML modellerini çalıştır."""
    print("\n" + "=" * 70)
    print("KLASİK ML PIPELINE")
    print("=" * 70)
    print(f"  Modeller: {CLASSICAL_MODELS}")
    print(f"  Pipeline: 64×64 grayscale → flatten → StandardScaler → PCA(0.95) → CLF")

    # 1) Raw pikselleri çıkar (cache varsa yükle)
    X, y = extract_raw_pixels(df, force=force)

    # 2) Her model için 3-fold CV
    for i, model_name in enumerate(CLASSICAL_MODELS, 1):
        trial_id = f"classical_{i:03d}"
        try:
            run_classical_trial(model_name, X, y, trial_id)
            print(f"\n✓ {trial_id} bitti.")
        except Exception as e:
            print(f"\n✗ {trial_id} hata verdi: {e}")
            import traceback; traceback.print_exc()


# =============================================================================
# ÖZET
# =============================================================================

def print_summary():
    if not MASTER_CSV.exists():
        print("⚠ all_trials.csv yok — henüz trial çalıştırılmadı.")
        return
    results_df = pd.read_csv(MASTER_CSV)
    if len(results_df) == 0:
        return

    print("\n" + "=" * 70)
    print(f"ÖZET — Toplam {len(results_df)} trial")
    print("=" * 70)

    # Kısa özet tablo
    show_cols = [c for c in [
        "trial_no", "model", "model_type", "augment",
        "loss_name", "optimizer_name", "scheduler_name",
        "train_accuracy_mean", "val_accuracy_mean",
        "train_f1_macro_mean", "val_f1_macro_mean",
        "wall_time_min",
    ] if c in results_df.columns]
    sub = results_df[show_cols].copy()
    print(sub.to_string(index=False))

    # En iyi
    if "val_f1_macro_mean" in results_df.columns:
        completed = results_df[results_df["status"] == "completed"] \
                    if "status" in results_df.columns else results_df
        if len(completed) > 0:
            best_idx = completed["val_f1_macro_mean"].idxmax()
            best = completed.loc[best_idx]
            trial_col = "trial_no" if "trial_no" in best.index else "trial_id"
            print(f"\n🏆 En iyi (VAL F1 macro):")
            print(f"   {best[trial_col]} | {best['model']}")
            print(f"   Train F1 = {best.get('train_f1_macro_mean', float('nan')):.4f} | "
                  f"Val F1 = {best['val_f1_macro_mean']:.4f} ± "
                  f"{best.get('val_f1_macro_std', 0):.4f}")
            print(f"   Train Acc = {best.get('train_accuracy_mean', float('nan')):.4f} | "
                  f"Val Acc = {best['val_accuracy_mean']:.4f} ± "
                  f"{best.get('val_accuracy_std', 0):.4f}")

    # Karşılaştırma grafiği
    plot_overall_comparison(results_df)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda", action="store_true", help="Sadece EDA")
    parser.add_argument("--deep", action="store_true",
                        help="Sadece deep Optuna search")
    parser.add_argument("--classical", action="store_true",
                        help="Sadece klasik ML")
    parser.add_argument("--force", action="store_true",
                        help="Cache'leri yok say (raw pixel yeniden çıkar)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search space'i listele, çalıştırma")
    args = parser.parse_args()

    run_all = not (args.eda or args.deep or args.classical or args.dry_run)

    set_all_seeds(RANDOM_SEED)
    print(f"Cihaz: {DEVICE}  |  Workers: {NUM_WORKERS}  |  AMP: {USE_MIXED_PRECISION}")
    print(f"Optuna: {'Available' if OPTUNA_AVAILABLE else 'NOT INSTALLED'}")
    print(f"MODEL_TYPES: {MODEL_TYPES}  →  ML: {'ML' in MODEL_TYPES} | DL: {'DL' in MODEL_TYPES}")

    df, label_map = load_metadata()
    print(f"Toplam {len(df)} görüntü yüklendi.")

    if run_all or args.eda:
        run_eda(df, label_map)

    # MODEL_TYPES filtresi: önce ML, sonra DL
    do_ml = ("ML" in MODEL_TYPES) and (run_all or args.classical)
    do_dl = ("DL" in MODEL_TYPES) and (run_all or args.deep or args.dry_run)

    # ML önce
    if do_ml and not args.dry_run:
        run_classical_pipeline(df, force=args.force)

    # DL sonra
    if do_dl:
        if not OPTUNA_AVAILABLE:
            print("\n⚠ Optuna yüklü değil — deep search atlanıyor.")
            print("   Yüklemek için: pip install optuna")
        else:
            run_optuna_study(df, dry_run=args.dry_run)

    print_summary()

    print("\n" + "=" * 70)
    print("✓ PIPELINE TAMAMLANDI")
    print(f"  Master CSV:  {MASTER_CSV}")
    print(f"  Sonuçlar:    {RESULTS_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
