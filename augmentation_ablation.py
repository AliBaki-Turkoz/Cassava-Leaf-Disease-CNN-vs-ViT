"""
augmentation_ablation.py

Train-time, on-the-fly augmentation ablation (Phase 2).
Merges the two original sweep scripts (augmentation_trial_ABT.py and
augmentation_trial_2.py), which differed only in their AUG_MODES list and
output directory.

=====================
CASSAVA — FAZ 2: AUGMENTATION SWEEP

Amaç:
  main.py (Faz 1) ile augmentation="none" kullanarak bulunan EN İYİ TRIAL'ın
  tüm hyperparameter ayarları sabit tutularak, AUG_MODES listesindeki modlar
  sırayla denenir. Hangi augmentation en iyi sonucu verdiği analiz edilir.

Kullanım:
  1. main.py'yi çalıştır → results/all_trials.csv'den en iyi trial'ı bul.
  2. Aşağıdaki BEST_TRIAL bölümüne en iyi trial'ın değerlerini gir.
  3. Bu dosyayı çalıştır:
       python augmentation_ablation.py                         # tüm modlar
       python augmentation_ablation.py --reported              # sadece makaledeki 15 mod
       python augmentation_ablation.py --aug heavy_combo,none  # seçili modlar
       python augmentation_ablation.py --dry-run               # sadece yapılandırmayı listele

Çıktılar (results_aug/ dizininde):
  - aug_trials.csv        : Her aug modu için tam metrikler (sütun sırası: trial_no,
                            model, augment, size, tüm parametreler, train metrikleri,
                            val metrikleri)
  - cm_aug_*.png/csv      : Confusion matrix
  - perclass_aug_*.png    : Sınıf bazlı precision/recall/F1 (Train + Val yan yana)
  - curves_aug_*.png      : Loss/accuracy training curves (fold'lar renkli)
  - report_aug_*.txt      : Train + Val classification_report
  - 99_aug_comparison.png : Tüm aug modlarının karşılaştırması (Train vs Val)
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
from PIL import Image as PILImage
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
    build_model,
    build_loss, build_optimizer, build_scheduler, is_plateau_scheduler,
    build_eval_transform,
    IMAGE_SIZE_DEEP, IMAGE_SIZE_VIT,
    IMAGENET_MEAN, IMAGENET_STD,
)


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           BEST TRIAL AYARLARI — BURAYA DOLDUR                          ║
# ║  main.py çalıştıktan sonra results/all_trials.csv'deki en iyi          ║
# ║  trial'ın değerlerini aşağıya kopyala.                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

# ---- Model ----
BEST_MODEL = "vit_b_16"           # "resnet50" | "efficientnet_b4" | "vit_b_16"

# ---- Learning rate ----
BEST_LR = 5e-4                    # örn: 1e-5 | 5e-5 | 1e-4 | 5e-4 | 1e-3

# ---- Weight decay ----
BEST_WEIGHT_DECAY = 1e-3          # örn: 1e-5 | 1e-4 | 1e-3 | 1e-2

# ---- Batch size ----
BEST_BATCH_SIZE = 16              # örn: 16 | 32 | 64

# ---- Loss ----
BEST_LOSS = "cross_entropy"       # "cross_entropy" | "cross_entropy_ls" | "focal" | "bi_tempered"

# Loss'a özel parametreler — kullanılan loss'a göre ilgili olanı doldur:
BEST_LABEL_SMOOTHING = 0.1        # cross_entropy_ls için
BEST_FOCAL_GAMMA = 2.0            # focal için
BEST_BI_T1 = 0.8                  # bi_tempered için
BEST_BI_T2 = 1.4                  # bi_tempered için
BEST_BI_LS = 0.06                 # bi_tempered label_smoothing için

# ---- Optimizer ----
BEST_OPTIMIZER = "sgd"            # "adamw" | "adam" | "sgd"
BEST_SGD_MOMENTUM = 0.95          # sadece sgd seçildiyse geçerli

# ---- Scheduler ----
BEST_SCHEDULER = "plateau"        # "plateau" | "cosine" | "cosine_warm"

# ---- Class weights ----
BEST_USE_CLASS_WEIGHTS = False    # True | False

# =============================================================================
# GENEL AYARLAR (main.py ile senkronize — DEĞİŞTİRME)
# =============================================================================

RANDOM_SEED = 42
NUM_CLASSES = 5
CLASS_NAMES_SHORT = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
CLASS_NAMES_FULL = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = Path(os.environ.get("CASSAVA_DATA_DIR", PROJECT_ROOT / "data"))
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TRAIN_CSV       = DATA_DIR / "train.csv"
LABEL_MAP_JSON  = DATA_DIR / "label_num_to_disease_map.json"

RESULTS_DIR     = PROJECT_ROOT / "results_aug"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints_aug"
for d in (RESULTS_DIR, CHECKPOINTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

AUG_CSV = RESULTS_DIR / "aug_trials.csv"

DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS         = min(4, os.cpu_count() or 1)
USE_MIXED_PRECISION = True
N_FOLDS             = 3   # main.py ile aynı

DEEP_EPOCHS             = 200   # main.py ile aynı
DEEP_EARLY_STOP_PATIENCE = 20   # main.py ile aynı
DEEP_GRAD_CLIP          = 1.0   # main.py ile aynı

# =============================================================================
# AUGMENTATION SWEEP LİSTESİ
# =============================================================================
# Tüm augmentation kodu bu dosyada. models.py'de augmentation YOKTUR.
#
# Kaggle Cassava Leaf Disease Classification yarışması araştırması:
#
#   1st Place (Golddiggaz, Private LB 91.32%):
#     ResNeXt50: RandomResizedCrop, Transpose, H/V Flip, ShiftScaleRotate, Normalize
#     ViT-B/16: + HueSaturationValue, RandomBrightnessContrast, CoarseDropout, Cutout
#     B4:       Flip, Transpose, Rotate, Saturation, Contrast, Brightness, random crop
#     NOT: "Hue tek başına iyileştirme getirmedi, az aug kullanmayı tercih ettik"
#
#   3rd Place (T0m, solo, Private LB 90.28%):
#     Sadece: RandomResizedCrop, Transpose, H/V Flip, ShiftScaleRotate, Normalize
#     Basit augmentation, overfitting endişesiyle ağır aug kullanmadı
#
#   14th Place (Kozodoi, Private LB 90.16%):
#     RandomResizedCrop, RandomGridShuffle, H/V Flip, Transpose,
#     ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast,
#     CLAHE, Cutout, Cutmix
#
# Sonuç: Cassava için EN ETKİLİ augmentation bileşenleri sırasıyla:
#   1. H/V Flip + Transpose (tüm çözümler kullandı)
#   2. RandomResizedCrop (tüm çözümler kullandı)
#   3. ShiftScaleRotate / RandomAffine (tüm çözümler kullandı)
#   4. ColorJitter (brightness/contrast/saturation) — 1st, 14th kullandı
#   5. Cutout / CoarseDropout / RandomErasing — 1st ViT, 14th kullandı
#   6. HueSaturationValue (hue shift) — 1st ViT, 14th kullandı (ama 1st tek başına yarar görmedi)
#   7. CLAHE — 14th kullandı (histogram eşitleme)
#   8. GridShuffle — 14th kullandı (yaprak dokusu karıştırma)
#
# Aşağıdaki modlar bu bilgiye dayalı olarak tasarlandı.
# =============================================================================

# ---------- Kategori A: Baseline ----------
# none                 → Augmentation yok (main.py reproduce kontrolü)
#
# ---------- Kategori B: Tekil Bileşen Ablasyonu ----------
# flip_only            → H/V Flip + Transpose (tek spatial)
# color_only           → ColorJitter + RandomGrayscale (tek renk)
# geometric_only       → RandomResizedCrop + Affine (tek geometri)
# erasing_only         → Flip + tek RandomErasing (Cutout etkisi)
# brightness_only      → Sadece brightness jitter
# contrast_only        → Sadece contrast jitter
# rotate_only          → Sadece rotation (flip yok, crop yok)
# blur_noise_light     → Hafif GaussianBlur (sensör gürültüsü simülasyonu)
# perspective_light    → Hafif perspektif bozulması
#
# ---------- Kategori C: Yarışma Reçeteleri ----------
# comp_resnext         → 1st place ResNeXt50 tam pipeline
# comp_vit             → 1st place ViT tam pipeline (en ağır)
# comp_3rd             → 3rd place basit ama etkili pipeline
#
# ---------- Kategori D: Kombine Seviyeler ----------
# light_combo          → Hafif: flip + transpose + hafif crop + hafif renk
# medium_combo         → Orta: crop + flip + affine + renk (önerilen default)
# heavy_combo          → Ağır: tam geometri + tam renk + erasing
# cutout_double        → Çift occlusion: Cutout + CoarseDropout
# clahe_enhanced       → 14th place'den ilham: CLAHE + renk + geometri
# safe_light_combo     → En güvenli hafif: crop(0.85)+flip+transpose+hafif renk
# geo_light_combo      → Hafif geometri: crop(0.8)+flip+hafif affine, renk yok
# color_light_combo    → Hafif renk: flip+ColorJitter, geometrik bozulma yok
# regularization_light_combo → Hafif regularization: crop+flip+hafif renk+hafif erasing

AUG_MODES = [
    # ---- Run 1 (originally augmentation_trial_ABT.py, output: results_aug/) ----
    # A: Baseline
    "none",
    
    # B: Tekil bileşen ablasyonu (4 adet)
    "contrast_only",
    "rotate_only",
    "blur_noise_light",
    "perspective_light",

    # C: Yarışma reçeteleri (2 adet — en ağır olanlar)
    "comp_resnext",
    "comp_vit",

    # D: Kombine seviyeler (5 adet)
    "heavy_combo",
    "cutout_double",
    "clahe_enhanced",
    "color_light_combo",
    "regularization_light_combo",

    # ---- Run 2 (originally augmentation_trial_2.py, output: results_aug_2/) ----
    # B: Tekil bileşen ablasyonu (5 adet)
    "flip_only",
    "color_only",
    "geometric_only",
    "erasing_only",
    "brightness_only",

    # C: Yarışma reçeteleri (1 adet — en hızlı)
    "comp_3rd",

    # D: Kombine seviyeler (4 adet)
    "light_combo",
    "medium_combo",
    "safe_light_combo",
    "geo_light_combo",
]

# The 15 policies whose results are reported in the paper (Table: augmentation
# ablation). The remaining policies above are defined and runnable, but no
# results for them are reported in the paper.
REPORTED_IN_PAPER = [
    "none", "contrast_only", "rotate_only", "blur_noise_light", "perspective_light",
    "comp_resnext", "comp_vit", "heavy_combo", "cutout_double", "clahe_enhanced",
    "color_light_combo", "regularization_light_combo",
    "comp_3rd", "flip_only", "color_only",
]

AUG_DESCRIPTIONS = {
    # A: Baseline
    "none":             "Sadece Resize+Normalize. Augmentation yok.",

    # B: Tekil bileşen ablasyonu
    "flip_only":        "H/V Flip + Transpose. Renk ve geometri yok.",
    "color_only":       "ColorJitter(b=0.3/c=0.3/s=0.3/hue=0.1) + Grayscale(p=0.05). Geometri yok.",
    "geometric_only":   "RandomResizedCrop(0.7-1.0) + RandomAffine(deg=20, t=10%, s=±15%). Renk yok.",
    "erasing_only":     "H/V Flip + Transpose + RandomErasing(p=0.5, 2-20%). Renk/geometri yok.",
    "brightness_only":  "Sadece brightness jitter (±0.3). Diğer renk/geometri yok.",
    "contrast_only":    "Sadece contrast jitter (±0.3). Diğer renk/geometri yok.",
    "rotate_only":      "Sadece RandomRotation(±20°). Flip/crop/renk yok.",
    "blur_noise_light": "Hafif GaussianBlur(σ=0.1-1.5). Sensör gürültüsü simülasyonu.",
    "perspective_light": "Hafif RandomPerspective(distortion=0.15). Açı farkı simülasyonu.",

    # C: Yarışma reçeteleri
    "comp_resnext":     "1st place ResNeXt50: RandomResizedCrop + Flip + Transpose + ShiftScaleRotate. Basit ama etkili.",
    "comp_vit":         "1st place ViT: Tam pipeline — crop+flip+affine+renk+hue+Cutout+CoarseDropout.",
    "comp_3rd":         "3rd place: RandomResizedCrop + Flip + Transpose + ShiftScaleRotate. Minimal ama gold.",

    # D: Kombine seviyeler
    "light_combo":      "Hafif: RandomResizedCrop(0.8-1.0) + Flip + Transpose + hafif ColorJitter(0.2).",
    "medium_combo":     "Orta: Crop(0.75-1.0) + Flip + Transpose + Affine(15°) + ColorJitter(0.25). Önerilen default.",
    "heavy_combo":      "Ağır: Crop(0.6-1.0) + Flip + Transpose + Affine(30°) + renk(hue) + RandomErasing.",
    "cutout_double":    "Çift occlusion: Orta geometri/renk + Cutout(p=0.5, 2-15%) + CoarseDropout(p=0.5, 1-8%).",
    "clahe_enhanced":   "14th place ilham: CLAHE + renk + geometri + hafif erasing. Histogram eşitleme ile kontrast.",
    "safe_light_combo":          "En güvenli hafif: Crop(0.85-1.0)+Flip+Transpose+ColorJitter(b=0.15,c=0.15). Minimum risk.",
    "geo_light_combo":           "Hafif geometri: Crop(0.8-1.0)+Flip+Affine(10°,t=5%,s=±5%). Renk yok.",
    "color_light_combo":         "Hafif renk: Flip+ColorJitter(b=0.2,c=0.2,s=0.1). Geometrik bozulma yok.",
    "regularization_light_combo": "Hafif reg: Crop(0.8-1.0)+Flip+ColorJitter(0.15)+RandomErasing(p=0.25,2-10%).",
}


# =============================================================================
# AUGMENTATION FACTORY
# Tüm augmentation pipeline'ları burada tanımlanır.
# models.py'de augmentation YOKTUR.
# =============================================================================

def _transpose_pil(img):
    """PIL Transpose (diagonal flip) — Pillow sürümüne göre güvenli."""
    try:
        return img.transpose(PILImage.Transpose.TRANSPOSE)  # Pillow >= 9.1
    except AttributeError:
        return img.transpose(PILImage.TRANSPOSE)            # Pillow < 9.1


def _clahe_pil(img, clip_limit=2.0, grid_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    14th place (Kozodoi) tarafından kullanıldı.
    Hastalık belirtilerinin kontrastını artırır — özellikle düşük
    ışık koşullarında ve homojen arka planlarda etkili.
    """
    import cv2
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def build_train_transform(aug_name: str, image_size: int = IMAGE_SIZE_DEEP):
    """
    Augmentation pipeline factory — Kaggle Cassava yarışmasına dayalı.

    Torchvision karşılıkları:
      Albumentations Transpose        → PIL.Image.Transpose.TRANSPOSE
      Albumentations ShiftScaleRotate → transforms.RandomAffine
      Albumentations HueSaturationValue + RandomBrightnessContrast
                                      → transforms.ColorJitter(hue=...)
      Albumentations CoarseDropout    → transforms.RandomErasing
      Albumentations CLAHE            → _clahe_pil (cv2 üzerinden)
    """

    # ===== A: BASELINE =====
    if aug_name == "none":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ===== B: TEKİL BİLEŞEN ABLASYONU =====
    elif aug_name == "flip_only":
        # Sadece flip + transpose — tüm yarışmacıların ortak minimumu
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "color_only":
        # Sadece renk jitter — geometri yok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "geometric_only":
        # Sadece geometri (crop + affine) — renk yok
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(
                degrees=20, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "erasing_only":
        # Sadece occlusion (flip + erasing) — renk ve crop yok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)),
        ])

    elif aug_name == "brightness_only":
        # Sadece brightness — diğer renk kanalları ve geometri yok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "contrast_only":
        # Sadece contrast — diğer renk kanalları ve geometri yok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "rotate_only":
        # Sadece rotation — flip, crop, renk yok
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "blur_noise_light":
        # Hafif Gaussian blur — sensör gürültüsü / odak kaybı simülasyonu
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "perspective_light":
        # Hafif perspektif bozulması — farklı açıdan çekim simülasyonu
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ===== C: YARIŞMA REÇETELERİ =====
    elif aug_name == "comp_resnext":
        # 1st place ResNeXt50 pipeline:
        #   RandomResizedCrop, Transpose, H/V Flip, ShiftScaleRotate, Normalize
        # Basit ama competition-proven — 512×512 orijinal, biz image_size kullanıyoruz
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(
                degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "comp_vit":
        # 1st place ViT tam pipeline:
        #   RandomResizedCrop, Transpose, H/V Flip, ShiftScaleRotate,
        #   HueSaturationValue, RandomBrightnessContrast, CoarseDropout, Cutout
        # En ağır yarışma pipeline'ı — ViT 384 için kullanıldı
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(
                degrees=25, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),   # Cutout
            transforms.RandomErasing(
                p=0.5, scale=(0.01, 0.08), ratio=(0.5, 2.0)),   # CoarseDropout
        ])

    elif aug_name == "comp_3rd":
        # 3rd place (T0m, solo gold) pipeline:
        #   RandomResizedCrop, Transpose(p=0.5), H/V Flip(p=0.5),
        #   ShiftScaleRotate(p=0.5), Normalize
        # Çok basit — overfitting korkusuyla minimal tutuldu ama gold aldı
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1)),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ===== D: KOMBİNE SEVİYELER =====
    elif aug_name == "light_combo":
        # Hafif kombine: flip + transpose + hafif crop + hafif renk
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "medium_combo":
        # Orta kombine: crop + flip + affine + renk (saturation dahil)
        # Önerilen default — dengeli risk/ödül
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "heavy_combo":
        # Ağır kombine: tam geometri + tam renk (hue) + erasing
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(
                degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
        ])

    elif aug_name == "cutout_double":
        # Çift occlusion: orta geometri/renk + iki RandomErasing katmanı
        # 1st place ViT'in CoarseDropout + Cutout stratejisi
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)),   # Cutout
            transforms.RandomErasing(
                p=0.5, scale=(0.01, 0.08), ratio=(0.5, 2.0)),   # CoarseDropout
        ])

    elif aug_name == "clahe_enhanced":
        # 14th place ilham: CLAHE histogram eşitleme + geometri + renk + hafif erasing
        # CLAHE, düşük kontrastlı yaprak görüntülerinde hastalık
        # belirtilerini öne çıkarır — özellikle CGM ve CBB için etkili
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(lambda img: _clahe_pil(img))], p=0.4),
            transforms.RandomAffine(
                degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.12)),
        ])

    # ===== D-2: HAFİF COMBO SEVİYELERİ =====
    elif aug_name == "safe_light_combo":
        # En güvenli hafif combo — crop(0.85)+flip+transpose+çok hafif renk
        # Minimum risk, baseline'dan sadece bir adım ötesi
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(_transpose_pil)], p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "geo_light_combo":
        # Hafif geometri combo — crop+flip+hafif affine, renk yok
        # Geometrik dönüşümlerin tek başına etkisini test eder
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "color_light_combo":
        # Hafif renk combo — flip+ColorJitter, geometrik bozulma yok
        # Crop/affine yok — görüntüyü geometrik bozmak istemediğiniz test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    elif aug_name == "regularization_light_combo":
        # Hafif regularization combo — crop+flip+hafif renk+hafif erasing
        # Erasing'in düşük dozda etkisini test eder
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)),
        ])

    else:
        raise ValueError(
            f"Bilinmeyen augmentation: '{aug_name}'.\n"
            f"Mevcut: {', '.join(AUG_MODES)}"
        )


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed: int = RANDOM_SEED):
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
# PARAMS BUILDER — BEST TRIAL ayarlarından params dict üret
# =============================================================================

def build_best_params(aug_name: str) -> dict:
    """
    BEST TRIAL bölümündeki sabit ayarlardan bir params dict oluşturur.
    Sadece augmentation değişir; geri kalan her şey sabit.
    """
    # Loss kwargs — seçilen loss'a göre
    loss_kwargs: dict = {}
    if BEST_LOSS == "cross_entropy_ls":
        loss_kwargs["label_smoothing"] = BEST_LABEL_SMOOTHING
    elif BEST_LOSS == "focal":
        loss_kwargs["gamma"] = BEST_FOCAL_GAMMA
    elif BEST_LOSS == "bi_tempered":
        loss_kwargs["t1"] = BEST_BI_T1
        loss_kwargs["t2"] = BEST_BI_T2
        loss_kwargs["label_smoothing"] = BEST_BI_LS

    # Optimizer kwargs — SGD seçilmişse momentum
    optimizer_kwargs: dict = {}
    if BEST_OPTIMIZER == "sgd":
        optimizer_kwargs["momentum"] = BEST_SGD_MOMENTUM
        optimizer_kwargs["nesterov"] = True

    # Scheduler kwargs — sabit (main.py ile aynı)
    scheduler_kwargs: dict = {}
    if BEST_SCHEDULER == "plateau":
        scheduler_kwargs = {"factor": 0.5, "patience": 2, "min_lr": 1e-7}
    elif BEST_SCHEDULER == "cosine":
        scheduler_kwargs = {"eta_min": 1e-7}
    elif BEST_SCHEDULER == "cosine_warm":
        scheduler_kwargs = {"T_0": 5, "T_mult": 2, "eta_min": 1e-7}

    return {
        "model":            BEST_MODEL,
        "lr":               BEST_LR,
        "weight_decay":     BEST_WEIGHT_DECAY,
        "batch_size":       BEST_BATCH_SIZE,
        "augmentation":     aug_name,
        "loss_name":        BEST_LOSS,
        "loss_kwargs":      loss_kwargs,
        "optimizer_name":   BEST_OPTIMIZER,
        "optimizer_kwargs": optimizer_kwargs,
        "scheduler_name":   BEST_SCHEDULER,
        "scheduler_kwargs": scheduler_kwargs,
        "use_class_weights": BEST_USE_CLASS_WEIGHTS,
    }


# =============================================================================
# DATASET
# =============================================================================

class CassavaDataset(Dataset):
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
# METRİKLER
# =============================================================================

def compute_metrics(y_true, y_pred):
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


def aggregate_fold_metrics(fold_metrics_list: list) -> dict:
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
# CSV ROW BUILDER
# Sütun sırası: trial_no, model, augment, size, aug_description,
#               tüm params/hyperparams, train metrikleri, val metrikleri
# =============================================================================

_AGG_METRIC_KEYS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
]


def _flatten_metrics(summary: dict, prefix: str) -> dict:
    out = {}
    if not summary:
        return out
    for k in _AGG_METRIC_KEYS:
        if f"{k}_mean" in summary:
            out[f"{prefix}_{k}_mean"] = round(summary[f"{k}_mean"], 4)
        if f"{k}_std" in summary:
            out[f"{prefix}_{k}_std"] = round(summary[f"{k}_std"], 4)
    for cls_idx, cls_name in enumerate(CLASS_NAMES_SHORT):
        for metric in ["precision", "recall", "f1"]:
            key_mean = f"per_class_{metric}_mean"
            key_std  = f"per_class_{metric}_std"
            if key_mean in summary:
                out[f"{prefix}_{metric}_{cls_name}_mean"] = round(
                    summary[key_mean][cls_idx], 4)
            if key_std in summary:
                out[f"{prefix}_{metric}_{cls_name}_std"] = round(
                    summary[key_std][cls_idx], 4)
    return out


def build_csv_row(trial_no: int, aug_name: str, params: dict,
                  val_summary: dict, train_summary: dict,
                  best_epochs: list, wall_time_sec: float,
                  status: str = "completed") -> dict:
    """
    CSV sütun sırası:
      1) trial_no, model, augment, size, aug_description
      2) Tüm parametreler ve hyperparametreler
      3) train_* metrikleri
      4) val_* metrikleri
    """
    image_size = IMAGE_SIZE_VIT if BEST_MODEL == "vit_b_16" else IMAGE_SIZE_DEEP

    row: dict = {}

    # 1) Kimlik
    row["trial_no"]        = trial_no
    row["model"]           = BEST_MODEL
    row["augment"]         = aug_name
    row["size"]            = image_size
    row["aug_description"] = AUG_DESCRIPTIONS.get(aug_name, "-")

    # 2) Tüm parametreler ve hyperparametreler
    row["timestamp"]          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["status"]             = status
    row["n_classes"]          = NUM_CLASSES
    row["n_folds"]            = N_FOLDS
    row["seed"]               = RANDOM_SEED
    row["device"]             = str(DEVICE)
    row["mixed_precision"]    = USE_MIXED_PRECISION and DEVICE.type == "cuda"
    row["epochs_max"]         = DEEP_EPOCHS
    row["early_stop_patience"] = DEEP_EARLY_STOP_PATIENCE
    row["grad_clip"]          = DEEP_GRAD_CLIP
    row["lr"]                 = params["lr"]
    row["weight_decay"]       = params["weight_decay"]
    row["batch_size"]         = params["batch_size"]
    row["loss_name"]          = params["loss_name"]
    row["loss_kwargs"]        = json.dumps(params.get("loss_kwargs", {}))
    row["optimizer_name"]     = params["optimizer_name"]
    row["optimizer_kwargs"]   = json.dumps(params.get("optimizer_kwargs", {}))
    row["scheduler_name"]     = params["scheduler_name"]
    row["scheduler_kwargs"]   = json.dumps(params.get("scheduler_kwargs", {}))
    row["use_class_weights"]  = params.get("use_class_weights", True)
    row["class_weight"]       = "balanced" if params.get("use_class_weights", True) else "none"
    row["best_epoch_per_fold"] = json.dumps([e for e in best_epochs if e is not None])
    row["wall_time_min"]      = round(wall_time_sec / 60, 2)

    # 3) Train metrikleri
    row.update(_flatten_metrics(train_summary, "train"))

    # 5) Val metrikleri
    row.update(_flatten_metrics(val_summary, "val"))

    return row


def append_to_aug_csv(row: dict):
    df_new = pd.DataFrame([row])
    new_cols = list(row.keys())
    if AUG_CSV.exists():
        df_existing = pd.read_csv(AUG_CSV)
        extra_cols = [c for c in df_existing.columns if c not in row]
        ordered_cols = new_cols + extra_cols
        df_existing = df_existing.reindex(columns=ordered_cols)
        df_new = df_new.reindex(columns=ordered_cols)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new[new_cols]
    df_combined.to_csv(AUG_CSV, index=False)


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================

def _subtitle(aug_name: str, params: dict, val_s: dict, train_s: dict) -> str:
    """Görselin altına yazılacak özet bilgi satırı."""
    val_f1  = round(val_s.get("f1_macro_mean", 0),  4) if val_s  else "-"
    val_acc = round(val_s.get("accuracy_mean", 0),   4) if val_s  else "-"
    tr_f1   = round(train_s.get("f1_macro_mean", 0), 4) if train_s else "-"
    tr_acc  = round(train_s.get("accuracy_mean", 0),  4) if train_s else "-"
    return (f"aug={aug_name} | model={BEST_MODEL} | "
            f"lr={params['lr']:.0e} | bs={params['batch_size']} | "
            f"loss={params['loss_name']} | opt={params['optimizer_name']} | "
            f"sch={params['scheduler_name']}\n"
            f"Train → acc={tr_acc}  f1_macro={tr_f1}   |   "
            f"Val → acc={val_acc}  f1_macro={val_f1}")


def plot_confusion_matrix(y_true, y_pred, aug_name: str,
                          params: dict, val_s: dict, train_s: dict,
                          trial_no: int):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Sayılar)")
    axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gerçek")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                ax=axes[1])
    axes[1].set_title("Normalize"); axes[1].set_xlabel("Tahmin")
    fig.suptitle(
        f"Trial {trial_no} | {BEST_MODEL} | aug={aug_name}\n"
        + _subtitle(aug_name, params, val_s, train_s),
        fontsize=9, y=1.02
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"cm_aug_{trial_no:02d}_{aug_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # CM csv
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES_SHORT,
                         columns=[f"pred_{c}" for c in CLASS_NAMES_SHORT])
    df_cm.index.name = "true_class"
    df_cm.to_csv(RESULTS_DIR / f"cm_aug_{trial_no:02d}_{aug_name}.csv")


def plot_per_class_metrics(val_s: dict, train_s: dict,
                           aug_name: str, params: dict,
                           val_summary: dict, train_summary: dict,
                           trial_no: int):
    """Train ve Val per-class precision/recall/F1 — yan yana 2 panel."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    x = np.arange(NUM_CLASSES); w = 0.25

    for ax, m, label in zip(axes, [train_s, val_s], ["TRAIN", "VALIDATION"]):
        if not m:
            ax.set_visible(False); continue
        p  = m["per_class_precision_mean"]
        r  = m["per_class_recall_mean"]
        f1 = m["per_class_f1_mean"]
        ax.bar(x - w, p,  w, label="Precision", color="#7F77DD")
        ax.bar(x,     r,  w, label="Recall",    color="#1D9E75")
        ax.bar(x + w, f1, w, label="F1",        color="#E24B4A")
        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES_SHORT)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Skor")
        ax.set_title(f"{label} — Sınıf Bazlı Performans ({N_FOLDS}-fold ort.)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(p):  ax.text(i - w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for i, v in enumerate(r):  ax.text(i,     v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for i, v in enumerate(f1): ax.text(i + w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle(
        f"Trial {trial_no} | {BEST_MODEL} | aug={aug_name}\n"
        + _subtitle(aug_name, params, val_summary, train_summary),
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"perclass_aug_{trial_no:02d}_{aug_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(histories: list, aug_name: str,
                         params: dict, val_summary: dict, train_summary: dict,
                         trial_no: int):
    """Loss ve Accuracy training curves — fold'lar renkli."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fold_idx, h in enumerate(histories, start=1):
        epochs = list(range(1, len(h["train_loss"]) + 1))
        axes[0].plot(epochs, h["train_loss"], "--", alpha=0.6,
                     label=f"Fold {fold_idx} train")
        axes[0].plot(epochs, h["val_loss"],   "-",  label=f"Fold {fold_idx} val")
        axes[1].plot(epochs, h["train_acc"],  "--", alpha=0.6,
                     label=f"Fold {fold_idx} train")
        axes[1].plot(epochs, h["val_acc"],    "-",  label=f"Fold {fold_idx} val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"aug={aug_name} — Loss"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"aug={aug_name} — Accuracy"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        f"Trial {trial_no} | {BEST_MODEL} | aug={aug_name}\n"
        + _subtitle(aug_name, params, val_summary, train_summary),
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"curves_aug_{trial_no:02d}_{aug_name}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_aug_comparison(results_df: pd.DataFrame):
    """Tüm aug modlarının Train vs Val karşılaştırma grafiği."""
    if len(results_df) == 0:
        return
    rdf = results_df[results_df["status"] == "completed"].copy()
    if len(rdf) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(rdf) * 1.2), 10))
    axes = axes.flatten()
    base_keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    lbls = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
    xpos = np.arange(len(rdf)); w = 0.35

    for ax, base, ml in zip(axes, base_keys, lbls):
        t_mean = f"train_{base}_mean"; t_std = f"train_{base}_std"
        v_mean = f"val_{base}_mean";   v_std = f"val_{base}_std"
        if t_mean in rdf.columns:
            ax.bar(xpos - w/2, rdf[t_mean].fillna(0), w,
                   yerr=rdf.get(t_std, pd.Series([0]*len(rdf))).fillna(0),
                   capsize=3, label="Train", color="#7F77DD",
                   edgecolor="black", linewidth=0.5)
        if v_mean in rdf.columns:
            ax.bar(xpos + w/2, rdf[v_mean].fillna(0), w,
                   yerr=rdf.get(v_std, pd.Series([0]*len(rdf))).fillna(0),
                   capsize=3, label="Val", color="#E24B4A",
                   edgecolor="black", linewidth=0.5)
        ax.set_title(ml); ax.set_ylabel(ml); ax.set_ylim(0, 1.05)
        ax.set_xticks(xpos)
        ax.set_xticklabels(rdf["augment"].values, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        # Değerleri bar üstüne yaz
        if v_mean in rdf.columns:
            for i, v in enumerate(rdf[v_mean].fillna(0)):
                ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

    plt.suptitle(
        f"Augmentation Karşılaştırması — {BEST_MODEL}\n"
        f"lr={BEST_LR:.0e} | bs={BEST_BATCH_SIZE} | "
        f"loss={BEST_LOSS} | opt={BEST_OPTIMIZER} | sch={BEST_SCHEDULER} | "
        f"cw={BEST_USE_CLASS_WEIGHTS}",
        fontsize=11, y=1.00
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "99_aug_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# EĞİTİM — TEK FOLD
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    total_loss = total_correct = total_n = 0
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
        total_loss    += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate_deep(model, loader, criterion, use_amp):
    model.eval()
    total_loss = total_correct = total_n = 0
    all_preds, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images); loss = criterion(logits, labels)
        else:
            logits = model(images); loss = criterion(logits, labels)
        total_loss    += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total_n       += images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (total_loss / total_n, total_correct / total_n,
            np.concatenate(all_preds), np.concatenate(all_labels))


def train_one_fold(params: dict, train_df, val_df, fold_idx: int):
    """Tek fold eğitimi — train ve val metriklerini birlikte döndürür."""
    set_all_seeds(RANDOM_SEED + fold_idx)

    img_size = IMAGE_SIZE_VIT if params["model"] == "vit_b_16" else IMAGE_SIZE_DEEP

    train_ds = CassavaDataset(train_df, TRAIN_IMAGES_DIR,
                              transform=build_train_transform(
                                  params["augmentation"], image_size=img_size))
    val_ds = CassavaDataset(val_df, TRAIN_IMAGES_DIR,
                            transform=build_eval_transform(image_size=img_size))

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

    model = build_model(params["model"], num_classes=NUM_CLASSES,
                        pretrained=True).to(DEVICE)

    if params.get("use_class_weights", True):
        cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES),
                                  y=train_df["label"].values)
        cw_tensor = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    else:
        cw_tensor = None

    criterion = build_loss(params["loss_name"], class_weights=cw_tensor,
                           **params["loss_kwargs"])
    optimizer = build_optimizer(params["optimizer_name"], model.parameters(),
                                lr=params["lr"], weight_decay=params["weight_decay"],
                                **params["optimizer_kwargs"])
    scheduler = build_scheduler(params["scheduler_name"], optimizer,
                                total_epochs=DEEP_EPOCHS, **params["scheduler_kwargs"])

    use_amp = USE_MIXED_PRECISION and DEVICE.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf"); best_state = None
    best_epoch = 0; patience_counter = 0

    print(f"\n  >> aug={params['augmentation']} | Fold {fold_idx+1}/{N_FOLDS} | "
          f"lr={params['lr']:.0e} bs={params['batch_size']} "
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
        history["val_loss"].append(val_loss);  history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if scheduler is not None:
            if is_plateau_scheduler(scheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        improved = val_loss < best_val_loss
        marker = "★" if improved else " "
        print(f"     Ep {epoch:3d} {marker} "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={current_lr:.1e} | {time.time()-t0:.1f}s")

        if improved:
            best_val_loss = val_loss; best_epoch = epoch
            best_state = deepcopy(model.state_dict()); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEEP_EARLY_STOP_PATIENCE:
                print(f"     ⚠ Early stop @ ep{epoch} (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Train eval (aug'suz — eval transform ile)
    train_eval_ds = CassavaDataset(
        train_df, TRAIN_IMAGES_DIR,
        transform=build_eval_transform(image_size=img_size))
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=params["batch_size"] * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    _, _, tr_preds, tr_true = evaluate_deep(model, train_eval_loader,
                                            criterion, use_amp)
    _, _, vl_preds, vl_true = evaluate_deep(model, val_loader, criterion, use_amp)

    train_m = compute_metrics(tr_true, tr_preds); train_m["best_epoch"] = best_epoch
    val_m   = compute_metrics(vl_true, vl_preds); val_m["best_epoch"]   = best_epoch

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_m, train_m, history, vl_preds, vl_true, tr_preds, tr_true


# =============================================================================
# EĞİTİM — N-FOLD CV (TEK AUG MODU)
# =============================================================================

def run_aug_trial(aug_name: str, trial_no: int, df: pd.DataFrame):
    """Bir augmentation modu için N-fold CV çalıştır, sonuçları kaydet."""
    params = build_best_params(aug_name)

    print(f"\n{'='*70}")
    print(f"AUG TRIAL {trial_no:02d} | aug={aug_name}")
    print(f"  {AUG_DESCRIPTIONS[aug_name]}")
    print(f"  model={BEST_MODEL} | lr={BEST_LR:.0e} | bs={BEST_BATCH_SIZE} | "
          f"loss={BEST_LOSS} | opt={BEST_OPTIMIZER} | sch={BEST_SCHEDULER} | "
          f"cw={BEST_USE_CLASS_WEIGHTS}")
    print(f"{'='*70}")

    t_start = time.time()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_all = df["label"].values; indices = np.arange(len(df))

    val_fold_m, train_fold_m, histories = [], [], []
    oof_preds = np.zeros(len(df), dtype=np.int64)
    oof_true  = np.zeros(len(df), dtype=np.int64)
    tr_true_all, tr_pred_all = [], []
    best_epochs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(indices, y_all)):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)
        v_m, t_m, h, v_preds, v_true, t_preds, t_true = train_one_fold(
            params, train_df, val_df, fold_idx)
        val_fold_m.append(v_m); train_fold_m.append(t_m)
        histories.append(h)
        oof_preds[va_idx] = v_preds; oof_true[va_idx] = v_true
        tr_true_all.append(t_true); tr_pred_all.append(t_preds)
        best_epochs.append(v_m["best_epoch"])
        print(f"     ⇒ Fold {fold_idx+1}: "
              f"train_acc={t_m['accuracy']:.4f} f1={t_m['f1_macro']:.4f} | "
              f"val_acc={v_m['accuracy']:.4f} f1={v_m['f1_macro']:.4f}")

    val_summary   = aggregate_fold_metrics(val_fold_m)
    train_summary = aggregate_fold_metrics(train_fold_m)
    tr_true_all   = np.concatenate(tr_true_all)
    tr_pred_all   = np.concatenate(tr_pred_all)
    wall = time.time() - t_start

    print(f"\n  ✓ {aug_name} bitti: "
          f"val_acc={val_summary['accuracy_mean']:.4f} "
          f"val_f1={val_summary['f1_macro_mean']:.4f} "
          f"({wall/60:.1f} dk)")

    # Görseller — her görselde trial no + aug + tüm metrikler
    plot_confusion_matrix(oof_true, oof_preds, aug_name,
                          params, val_summary, train_summary, trial_no)
    plot_per_class_metrics(val_summary, train_summary,
                           aug_name, params,
                           val_summary, train_summary, trial_no)
    plot_training_curves(histories, aug_name, params,
                         val_summary, train_summary, trial_no)

    # Report
    report_val   = classification_report(oof_true, oof_preds,
                                         target_names=CLASS_NAMES_SHORT,
                                         digits=4, zero_division=0)
    report_train = classification_report(tr_true_all, tr_pred_all,
                                         target_names=CLASS_NAMES_SHORT,
                                         digits=4, zero_division=0)
    with open(RESULTS_DIR / f"report_aug_{trial_no:02d}_{aug_name}.txt",
              "w", encoding="utf-8") as f:
        f.write(f"=== Trial {trial_no:02d} | aug={aug_name} ===\n\n"
                f"Model:     {BEST_MODEL}\n"
                f"Aug:       {aug_name}  —  {AUG_DESCRIPTIONS[aug_name]}\n"
                f"lr:        {BEST_LR}\n"
                f"wd:        {BEST_WEIGHT_DECAY}\n"
                f"batch:     {BEST_BATCH_SIZE}\n"
                f"loss:      {BEST_LOSS}  {params['loss_kwargs']}\n"
                f"optimizer: {BEST_OPTIMIZER}  {params['optimizer_kwargs']}\n"
                f"scheduler: {BEST_SCHEDULER}  {params['scheduler_kwargs']}\n"
                f"cw:        {BEST_USE_CLASS_WEIGHTS}\n\n"
                f"--- TRAIN ({N_FOLDS}-fold) ---\n{report_train}\n\n"
                f"--- VALIDATION ({N_FOLDS}-fold OOF) ---\n{report_val}\n")

    # CSV
    row = build_csv_row(trial_no, aug_name, params,
                        val_summary, train_summary, best_epochs, wall,
                        status="completed")
    append_to_aug_csv(row)

    # Checkpoint (son fold modeli değil — sadece study kayıt referansı)
    joblib.dump({"aug": aug_name, "params": params,
                 "val_summary": val_summary, "train_summary": train_summary},
                CHECKPOINTS_DIR / f"aug_trial_{trial_no:02d}_{aug_name}.pkl")
    return row


# =============================================================================
# ÖZET
# =============================================================================

def print_summary():
    if not AUG_CSV.exists():
        print("⚠ aug_trials.csv yok — henüz trial çalıştırılmadı."); return
    rdf = pd.read_csv(AUG_CSV)
    if len(rdf) == 0: return

    print("\n" + "=" * 70)
    print(f"AUG SWEEP ÖZET — {BEST_MODEL} | {len(rdf)} aug modu")
    print("=" * 70)

    show_cols = [c for c in [
        "trial_no", "augment",
        "train_accuracy_mean", "train_f1_macro_mean",
        "val_accuracy_mean",   "val_f1_macro_mean",
        "delta_val_f1", "wall_time_min",
    ] if c in rdf.columns]
    print(rdf[show_cols].to_string(index=False))

    completed = rdf[rdf["status"] == "completed"] if "status" in rdf.columns else rdf
    if len(completed) > 0 and "val_f1_macro_mean" in completed.columns:
        best_idx = completed["val_f1_macro_mean"].idxmax()
        best = completed.loc[best_idx]
        print(f"\n🏆 En iyi augmentation: '{best['augment']}'")
        print(f"   Val F1  = {best['val_f1_macro_mean']:.4f}")
        print(f"   Val Acc = {best['val_accuracy_mean']:.4f}")
        print(f"   Train F1  = {best.get('train_f1_macro_mean', float('nan')):.4f}")
        print(f"   Train Acc = {best.get('train_accuracy_mean', float('nan')):.4f}")

    plot_aug_comparison(rdf)
    print(f"\n✓ Karşılaştırma grafiği: {RESULTS_DIR / '99_aug_comparison.png'}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cassava Faz 2 — Augmentation Sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece yapılandırmayı listele, çalıştırma")
    parser.add_argument("--aug", type=str, default=None,
                        help="Virgülle ayrılmış aug modları (örn: --aug heavy_combo,none)")
    parser.add_argument("--reported", action="store_true",
                        help="Sadece makalede raporlanan 15 modu çalıştır")
    args = parser.parse_args()

    set_all_seeds(RANDOM_SEED)
    print(f"Cihaz: {DEVICE}  |  Workers: {NUM_WORKERS}  |  AMP: {USE_MIXED_PRECISION}")
    print(f"\n{'='*70}")
    print("AUGMENTATION SWEEP — FAZ 2 AYARLARI")
    print(f"{'='*70}")
    print(f"  Model:          {BEST_MODEL}")
    print(f"  LR:             {BEST_LR}")
    print(f"  Weight decay:   {BEST_WEIGHT_DECAY}")
    print(f"  Batch size:     {BEST_BATCH_SIZE}")
    print(f"  Loss:           {BEST_LOSS}")
    print(f"  Optimizer:      {BEST_OPTIMIZER}")
    print(f"  Scheduler:      {BEST_SCHEDULER}")
    print(f"  Class weights:  {BEST_USE_CLASS_WEIGHTS}")
    print(f"  Folds:          {N_FOLDS}")
    print(f"  Epochs (max):   {DEEP_EPOCHS}  (early stop: {DEEP_EARLY_STOP_PATIENCE})")
    print(f"\n  Augmentation modları ({len(AUG_MODES)}):")
    for i, aug in enumerate(AUG_MODES):
        print(f"    {i+1}. {aug:12s} — {AUG_DESCRIPTIONS[aug]}")

    if args.dry_run:
        print("\n--dry-run: çalıştırılmadı.")
        return

    df, _ = load_metadata()
    print(f"\nToplam {len(df)} görüntü yüklendi.")

    # Çalıştırılacak aug listesi
    if args.aug:
        run_modes = [m.strip() for m in args.aug.split(",") if m.strip()]
    elif args.reported:
        run_modes = list(REPORTED_IN_PAPER)
    else:
        run_modes = list(AUG_MODES)
    unknown = [m for m in run_modes if m not in AUG_MODES]
    if unknown:
        raise SystemExit(f"Bilinmeyen augmentation modu: {unknown}")

    for trial_no, aug_name in enumerate(run_modes, start=1):
        try:
            run_aug_trial(aug_name, trial_no, df)
        except Exception as e:
            print(f"\n✗ aug={aug_name} hata verdi: {e}")
            import traceback; traceback.print_exc()
            # Hatalı trial'ı CSV'ye yaz
            params = build_best_params(aug_name)
            row = build_csv_row(trial_no, aug_name, params, {}, {},
                                [], 0, status=f"failed: {str(e)[:100]}")
            append_to_aug_csv(row)

    print_summary()

    print("\n" + "=" * 70)
    print("✓ AUGMENTATION SWEEP TAMAMLANDI")
    print(f"  CSV:     {AUG_CSV}")
    print(f"  Sonuçlar: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
