"""
models.py
=========
MODELLER + FACTORY'LER

İçerik:
  Deep modeller: resnet50, efficientnet_b4, vit_b_16
  Klasik ML:     random_forest, extra_trees, logistic_regression, sgd_classifier (raw pixel + PCA)
  Loss factory:  cross_entropy / cross_entropy_ls / focal / bi_tempered
  Optimizer:     adamw / adam / sgd / rmsprop
  Scheduler:     plateau / cosine / cosine_warm / exponential / step / none
  Eval transform: Resize + Normalize (validation/test için)

NOT: Augmentation kodları augmentation_ablation.py içindedir.
     main.py (Faz 1) sadece augmentation="none" kullanır → build_eval_transform yeterli.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm, transforms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight


# =========================================================================
# SABİTLER
# =========================================================================
RANDOM_SEED = 42
NUM_CLASSES = 5
IMAGE_SIZE = 224                    # deep modeller için (legacy)
IMAGE_SIZE_DEEP = 224               # ResNet50 / EfficientNet-B4 için
IMAGE_SIZE_VIT = 384                # ViT-B/16 için — 384 önerilen boyut
IMAGE_SIZE_CLASSICAL = 64           # klasik ML: 64×64 grayscale flatten
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================================
# MODEL TİPİ KAYITI
# =========================================================================
MODEL_TYPE = {
    "resnet50":             "deep",
    "efficientnet_b4":      "deep",
    "vit_b_16":             "deep",
    "random_forest":        "classical",
    "extra_trees":          "classical",
    "logistic_regression":  "classical",
    "sgd_classifier":       "classical",
}
AVAILABLE_MODELS = list(MODEL_TYPE.keys())


# =========================================================================
# DEEP MODELLER
# =========================================================================

def _build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet50 — Cassava için baseline. ~25M parametre."""
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_efficientnet_b4(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 — compound scaling ile daha güçlü. ~19M parametre."""
    weights = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _build_vit_b_16(num_classes: int, pretrained: bool = True,
                    image_size: int = IMAGE_SIZE_VIT) -> nn.Module:
    """
    ViT-Base/16 — transformer based. ~86M parametre.

    image_size seçenekleri:
      - 224: ImageNet varsayılanı (daha hızlı, daha düşük performans)
      - 384: Kaggle Cassava 1. place; patch grid 24×24 → daha iyi özellik

    NOT: 384 boyut için IMAGENET1K_SWAG_E2E_V1 weights önerilir —
         bu weights 384 resolution üzerinde fine-tune edilmiştir.
    """
    if image_size == 384:
        # 384×384 için SWAG weights (IMAGENET1K_SWAG_E2E_V1) daha uygun
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        model = tvm.vit_b_16(weights=weights, image_size=384)
    else:
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


_DEEP_BUILDERS = {
    "resnet50":         _build_resnet50,
    "efficientnet_b4":  _build_efficientnet_b4,
    "vit_b_16":         _build_vit_b_16,
}


def get_deep_model(name: str, num_classes: int = NUM_CLASSES,
                   pretrained: bool = True) -> nn.Module:
    """Deep model döndür — tüm parametreler trainable (full fine-tuning).

    ViT-B/16 için otomatik olarak IMAGE_SIZE_VIT (384) kullanılır;
    diğer modeller IMAGE_SIZE_DEEP (224) kullanır.
    """
    if name not in _DEEP_BUILDERS:
        raise ValueError(f"Bilinmeyen deep model: '{name}'")
    if name == "vit_b_16":
        model = _build_vit_b_16(num_classes=num_classes, pretrained=pretrained,
                                image_size=IMAGE_SIZE_VIT)
    else:
        model = _DEEP_BUILDERS[name](num_classes=num_classes, pretrained=pretrained)
    for p in model.parameters():
        p.requires_grad = True
    return model


# =========================================================================
# KLASİK ML MODELLERİ — Raw pixel + StandardScaler + PCA + Classifier
# =========================================================================

def _make_classical_pipeline(classifier, pca_variance: float = 0.95) -> Pipeline:
    """
    Klasik ML için standart pipeline:
        Raw flatten pixels -> StandardScaler -> PCA -> Classifier

    Sıra önemli:
      1. StandardScaler: PCA için zorunlu (mean=0, std=1)
      2. PCA: 4096-dim → ~150-300-dim (varyans oranına göre)
      3. Classifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_variance, random_state=RANDOM_SEED)),
        ("clf", classifier),
    ])


def get_random_forest(pca_variance: float = 0.95) -> Pipeline:
    """Random Forest — n_estimators=300, class_weight='balanced'."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return _make_classical_pipeline(rf, pca_variance=pca_variance)


def get_extra_trees(pca_variance: float = 0.95) -> Pipeline:
    """Extra Trees — RF'ye benzer ama split noktaları rastgele → daha hızlı."""
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return _make_classical_pipeline(et, pca_variance=pca_variance)


def get_logistic_regression(pca_variance: float = 0.95) -> Pipeline:
    """Logistic Regression — multinomial, class_weight='balanced'."""
    lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return _make_classical_pipeline(lr, pca_variance=pca_variance)


def get_sgd_classifier(pca_variance: float = 0.95) -> Pipeline:
    """SGD Classifier — log loss, class_weight='balanced'. Hızlı lineer model."""
    sgd = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return _make_classical_pipeline(sgd, pca_variance=pca_variance)


_CLASSICAL_BUILDERS = {
    "random_forest":        get_random_forest,
    "extra_trees":          get_extra_trees,
    "logistic_regression":  get_logistic_regression,
    "sgd_classifier":       get_sgd_classifier,
}


def get_classical_model(name: str, pca_variance: float = 0.95) -> Pipeline:
    if name not in _CLASSICAL_BUILDERS:
        raise ValueError(f"Bilinmeyen klasik model: '{name}'")
    return _CLASSICAL_BUILDERS[name](pca_variance=pca_variance)


# =========================================================================
# UNIFIED ARAYÜZ
# =========================================================================

def build_model(name: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = True, pca_variance: float = 0.95):
    """İsme göre uygun modeli döndür."""
    if name not in MODEL_TYPE:
        raise ValueError(f"Bilinmeyen model: '{name}'")
    if MODEL_TYPE[name] == "deep":
        return get_deep_model(name, num_classes=num_classes, pretrained=pretrained)
    return get_classical_model(name, pca_variance=pca_variance)


def fit_classical_model(name: str, model: Pipeline,
                        X_train: np.ndarray, y_train: np.ndarray):
    """Klasik modeli eğit."""
    model.fit(X_train, y_train)
    return model


def model_info(name: str) -> dict:
    info = {"name": name, "type": MODEL_TYPE.get(name, "unknown")}
    if MODEL_TYPE.get(name) == "deep":
        m = get_deep_model(name, pretrained=False)
        n_params = sum(p.numel() for p in m.parameters())
        info["parameters_M"] = round(n_params / 1e6, 2)
        del m
    return info


# =========================================================================
# LOSS FACTORY
# =========================================================================


class BiTemperedLogisticLoss(nn.Module):
    """
    Bi-Tempered Logistic Loss (Amid et al. 2019).

    Kaggle Cassava için neden kritik?
      - Dataset GÜRÜLTÜLÜ etiketler içeriyor (hem train hem test)
      - Standart CE, gürültülü örneklere büyük gradient vererek overfit eder
      - Bi-Tempered loss: outlier'ları downweight eder (bounded loss)
      - 1. place: t1=0.8, t2=1.4 kullandı
      - Referans: https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss.html

    Parametreler:
      t1 (0 < t1 < 1): küçük probabilite logit'leri sınırlar (noisy sample kontrolü)
      t2 (t2 > 1):     büyük loss değerlerini sınırlar (outlier kontrolü)
      t1=1, t2=1 → standart cross-entropy'ye eşdeğer
    """
    def __init__(self, t1: float = 0.8, t2: float = 1.4,
                 label_smoothing: float = 0.0,
                 num_iters: int = 5):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    def _log_t(self, u, t):
        """Tempered log: log_t(u) = (u^(1-t) - 1) / (1 - t)"""
        if t == 1.0:
            return torch.log(u)
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)

    def _exp_t(self, u, t):
        """Tempered exp: exp_t(u) = [max(0, 1 + (1-t)*u)]^(1/(1-t))"""
        if t == 1.0:
            return torch.exp(u)
        return torch.clamp(1.0 + (1.0 - t) * u, min=0.0).pow(1.0 / (1.0 - t))

    def _compute_normalization(self, activations, t):
        """Normalizasyon sabitini iteratif olarak hesapla."""
        mu = activations.max(dim=-1, keepdim=True).values
        effective_dim = activations.shape[-1]
        z = torch.full_like(mu, 1.0 / effective_dim)
        for _ in range(self.num_iters):
            z = self._exp_t(activations - mu, t)
            z_sum = z.sum(dim=-1, keepdim=True)
            mu = self._log_t(z_sum / effective_dim, t) + mu
        return -self._log_t(1.0 / effective_dim, t) + mu

    def forward(self, logits, targets):
        num_classes = logits.shape[-1]
        # Label smoothing
        if self.label_smoothing > 0.0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = targets_one_hot * (
                1.0 - self.label_smoothing
            ) + self.label_smoothing / num_classes
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()

        # Normalization
        z = self._compute_normalization(logits, self.t2)
        # Tempered softmax probabilities
        probs = self._exp_t(logits - z, self.t2)
        # Bi-tempered loss
        loss = (
            targets_one_hot * self._log_t(targets_one_hot + 1e-10, self.t1)
            - targets_one_hot * self._log_t(probs, self.t1)
            - (1.0 / (2.0 - self.t1))
            * (targets_one_hot.pow(2.0 - self.t1) - probs.pow(2.0 - self.t1))
        ).sum(dim=-1)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017): FL = -alpha * (1-p)^gamma * log(p)

    Cassava için faydası:
      - CMD %61 baskın → modeli çoğu zaman doğru tahmin ediyor → "kolay" örnek
      - gamma yüksek olunca bu kolay örnekler downweight olur
      - Az örnekli sınıflara (CBB %5) odaklanma artar
      - gamma=0 → standart cross-entropy
      - gamma=2 → varsayılan (en yaygın)
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(loss_name: str, class_weights: Optional[torch.Tensor] = None,
               **kwargs) -> nn.Module:
    """
    Loss factory.

    Mevcut:
      - 'cross_entropy'    : standart CE + class_weight
      - 'cross_entropy_ls' : label smoothing (param: label_smoothing)
                              → overconfident tahminleri yumuşatır
      - 'focal'            : focal loss (param: gamma)
                              → dengesiz veri için odaklı eğitim
      - 'bi_tempered'      : Bi-Tempered Logistic Loss (param: t1, t2, label_smoothing)
                              → GÜRÜLTÜLÜ ETİKETLER için en etkili yöntem
                              → Kaggle Cassava 1. ve 3. place tarafından kullanıldı
                              → Önerilen: t1=0.8, t2=1.4, label_smoothing=0.06
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "cross_entropy_ls":
        ls = kwargs.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ls)
    elif loss_name == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)
    elif loss_name == "bi_tempered":
        t1 = kwargs.get("t1", 0.8)
        t2 = kwargs.get("t2", 1.4)
        ls = kwargs.get("label_smoothing", 0.06)
        return BiTemperedLogisticLoss(t1=t1, t2=t2, label_smoothing=ls)
    else:
        raise ValueError(f"Bilinmeyen loss: '{loss_name}'")


# =========================================================================
# OPTIMIZER FACTORY
# =========================================================================

def build_optimizer(opt_name: str, params, lr: float, weight_decay: float = 0.0,
                    **kwargs) -> torch.optim.Optimizer:
    """
    Optimizer factory.

    Cassava için önerilen sıralama:
      1. AdamW   → en yaygın, hızlı yakınsama, weight_decay düzgün
      2. Adam    → AdamW'nin atası
      3. SGD     → yavaş ama bazen daha iyi generalize, momentum=0.9 standart
      4. RMSprop → CNN için nadir kullanılır
    """
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", True)
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               nesterov=nesterov, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        momentum = kwargs.get("momentum", 0.9)
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum,
                                   weight_decay=weight_decay)
    else:
        raise ValueError(f"Bilinmeyen optimizer: '{opt_name}'")


# =========================================================================
# SCHEDULER FACTORY
# =========================================================================

def build_scheduler(sched_name: str, optimizer, total_epochs: int = 30,
                    **kwargs):
    """
    LR Scheduler factory.

    Cassava için önerilenler:
      - 'plateau'      → val_loss durgunlaşınca düşür (en yaygın)
      - 'cosine'       → smooth azalma (Kaggle Cassava 1st place)
      - 'cosine_warm'  → periyodik restart, local minima'dan kurtulur
    """
    if sched_name == "none":
        return None
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 2),
            min_lr=kwargs.get("min_lr", 1e-7))
    elif sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=kwargs.get("eta_min", 1e-7))
    elif sched_name == "cosine_warm":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 5),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-7))
    elif sched_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95))
    elif sched_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.5))
    else:
        raise ValueError(f"Bilinmeyen scheduler: '{sched_name}'")


def is_plateau_scheduler(scheduler) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


# =========================================================================
# TRANSFORM (EVAL / CLASSICAL)
# =========================================================================

def build_eval_transform(image_size: int = IMAGE_SIZE_DEEP):
    """Validation/test için sabit transform — augmentation YOK."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_classical_transform(image_size: int = IMAGE_SIZE_CLASSICAL):
    """
    Klasik ML modelleri için transform.

    Pipeline:
      1. Resize → image_size × image_size (64×64)
      2. Grayscale → tek kanal (renkten bağımsız doku özellikleri)
      3. ToTensor → [0, 1] float
      4. Flatten → (1, H, W) → (H*W,) vektör — Pipeline'da numpy'e çevrilir

    NOT: Bu transform CassavaDataset ile kullanıldığında
         __getitem__ tensor döndürür (1, 4096). main.py'de
         .view(batch_size, -1) ile flatten edilmeli.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),             # (1, H, W), float [0,1]
        transforms.Lambda(lambda x: x.view(-1)),  # (H*W,) flatten
    ])


# =========================================================================
# HIZLI TEST
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FACTORY TESTLERİ")
    print("=" * 60)

    print("\n[Modeller]")
    for name in AVAILABLE_MODELS:
        info = model_info(name)
        suffix = f" ({info.get('parameters_M','?')}M)" if info["type"] == "deep" else ""
        print(f"  {name:18s} {info['type']}{suffix}")

    print("\n[Loss factory]")
    for ln in ["cross_entropy", "cross_entropy_ls", "focal", "bi_tempered"]:
        loss = build_loss(ln, label_smoothing=0.1, gamma=2.0)
        print(f"  {ln:20s} {type(loss).__name__}")

    print("\n[Optimizer factory]")
    dummy = nn.Linear(10, 5)
    for on in ["adamw", "adam", "sgd", "rmsprop"]:
        opt = build_optimizer(on, dummy.parameters(), lr=1e-3, weight_decay=1e-4)
        print(f"  {on:10s} {type(opt).__name__}")

    print("\n[Scheduler factory]")
    opt = torch.optim.AdamW(dummy.parameters(), lr=1e-3)
    for sn in ["none", "plateau", "cosine", "cosine_warm", "exponential", "step"]:
        sch = build_scheduler(sn, opt, total_epochs=30)
        print(f"  {sn:14s} {type(sch).__name__ if sch else 'None'}")

    print("\n[Eval transform]")
    et = build_eval_transform(IMAGE_SIZE_DEEP)
    print(f"  eval_transform: {len(et.transforms)} steps")

    print("\n✓ Tüm factory'ler çalışıyor.")
