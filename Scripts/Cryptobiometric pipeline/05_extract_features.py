"""
05_extract_features.py
Entrenamiento de extractor de características para iris biométrico.
Arquitectura: EfficientNetV2-S + ArcFace loss
 
Modos de entrenamiento disponibles (--mode):
  - joint  : Entrenamiento conjunto con todos los datasets
             (CASIA + IITD + PolyU NIR + PolyU VIS + UBIRIS).
             Clases separadas por espectro en PolyU para evitar alta
             varianza intraclase. Modelo guardado en joint.pth.
             → Usado en exp07: EER final = 0.86%
 
  - phase1 : Entrenamiento solo con NIR (CASIA + IITD).
             Modelo guardado en phase1.pth.
             → Usado en exp06: EER final NIR = 0.88%
 
  - phase2 : Fine-tuning VIS sobre phase1 (PolyU VIS + UBIRIS).
             Modelo guardado en phase2.pth.
 
  - extract: Carga un modelo ya entrenado, evalúa y guarda embeddings.
 
Split 70/30 por sujeto:
  - 70% sujetos → desarrollo (80% train / 20% val, monitoreo EER par a par)
  - 30% sujetos → evaluación (50% imágenes enrolamiento / 50% test)
  Evaluación final con protocolo score máximo exhaustivo sobre sujetos nuevos.
 
Uso:
    # Entrenamiento conjunto — todos los datasets (exp07)
    python scripts/05_extract_features.py --run_name exp07 --mode joint \
        --epochs_phase1 50 --lr_phase1 1e-4 --margin_phase1 0.4 \
        --warmup_phase1 5 --early_stopping 15 --eval_every 5
 
    # Entrenamiento NIR (exp06 fase 1)
    python scripts/05_extract_features.py --run_name exp06 --mode phase1
 
    # Fine-tuning VIS (exp06 fase 2)
    python scripts/05_extract_features.py --run_name exp06 --mode phase2
 
    # Solo extraer embeddings con modelo ya entrenado
    python scripts/05_extract_features.py --run_name exp07 --mode extract --phase joint
"""

import cv2
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import timm
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# 2. WARMUP + COSINE SCHEDULER
# ─────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Warmup lineal + decay coseno.
    Soporta discriminative learning rates: escala cada grupo
    proporcionalmente a su lr inicial definido en el optimizer.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 lr_max, lr_min=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr_max        = lr_max
        self.lr_min        = lr_min
        self.current_epoch = 0
        # Guardar lr base de cada grupo en el constructor
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Warmup lineal: de lr_min hasta lr_max
            lr = self.lr_min + (self.lr_max - self.lr_min) *                  self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay: de lr_max hasta lr_min
            progress = (self.current_epoch - self.warmup_epochs) /                        (self.total_epochs  - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) *                  (1 + math.cos(math.pi * progress))

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr


# ─────────────────────────────────────────────
# 3. ARCFACE LOSS
# ─────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes,
                 scale=64.0, margin=0.5, label_smoothing=0.0):
        super().__init__()
        self.scale           = scale
        self.margin          = margin
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        emb_norm    = nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        cosine = nn.functional.linear(emb_norm, weight_norm)
        sine   = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return nn.functional.cross_entropy(output, labels,
                                           label_smoothing=self.label_smoothing)


# ─────────────────────────────────────────────
# 4. MODELO: EfficientNetV2-S ADAPTADO
# ─────────────────────────────────────────────

class IrisEfficientNet(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=True, dropout=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        original_conv = self.backbone.conv_stem
        self.backbone.conv_stem = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        with torch.no_grad():
            self.backbone.conv_stem.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        in_features = self.backbone.num_features
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, embedding_dim))
        layers.append(nn.BatchNorm1d(embedding_dim))
        self.embedding = nn.Sequential(*layers)

    def forward(self, x):
        features  = self.backbone(x)
        embedding = self.embedding(features)
        return nn.functional.normalize(embedding, p=2, dim=1)

    def freeze_backbone(self, unfreeze_last_n_blocks=0):
        """
        Congela el backbone completo excepto los últimos n bloques.
        unfreeze_last_n_blocks=0 → congela todo
        unfreeze_last_n_blocks=2 → descongela los 2 últimos bloques
        """
        # Congelar todo primero
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # Descongelar los últimos N bloques
            blocks = list(self.backbone.blocks)
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # También descongelar conv_head y bn2 (capas finales del backbone)
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in self.backbone.parameters()
                            if p.requires_grad)
            print(f"  ✓ Backbone parcialmente congelado — "
                  f"últimos {unfreeze_last_n_blocks} bloques + conv_head "
                  f"entrenables ({trainable/1e6:.1f}M params)")
        else:
            print("  ✓ Backbone completamente congelado")


# ─────────────────────────────────────────────
# 5. DATASET
# ─────────────────────────────────────────────

class IrisDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples      = samples
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_str = self.samples[idx]
        label = self.class_to_idx[class_str]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 512), dtype=np.uint8)
        img = torch.FloatTensor(img).unsqueeze(0) / 255.0
        if self.transform:
            img = self.transform(img)
        return img, label



def build_dataset_joint(polar_root, **kwargs):
    """
    Entrenamiento conjunto: CASIA + IITD + PolyU NIR + PolyU VIS + UBIRIS.
    Clases separadas por espectro en PolyU para evitar alta varianza intraclase.
    Sujetos de PolyU agrupados (NIR+VIS juntos) para evitar data leakage.

    Clases:
      casia_{subject}_{eye}
      iitd_{subject}_{eye}
      polyu_{subject}_{eye}_NIR
      polyu_{subject}_{eye}_VIS
      ubiris_{eye_id}
    """
    polar_root    = Path(polar_root)
    class_samples = defaultdict(list)

    # CASIA
    for img_path in sorted(polar_root.glob("casia/*/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        subject = img_path.parts[-3]
        eye     = img_path.parts[-2]
        class_samples[f"casia_{subject}_{eye}"].append(img_path)

    # IITD
    for img_path in sorted(polar_root.glob("iitd/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        subject    = img_path.parts[-2]
        parts_name = img_path.stem.replace("_polar", "").split("_")
        eye        = parts_name[1] if len(parts_name) > 1 else "L"
        class_samples[f"iitd_{subject}_{eye}"].append(img_path)

    # PolyU NIR — clase con espectro
    for img_path in sorted(polar_root.glob("polyu/*/*/NIR/*polar.png")):
        if "_mask" in img_path.name:
            continue
        subject = img_path.parts[-4]
        eye     = img_path.parts[-3]
        class_samples[f"polyu_{subject}_{eye}_NIR"].append(img_path)

    # PolyU VIS — clase con espectro
    for img_path in sorted(polar_root.glob("polyu/*/*/VIS/*polar.png")):
        if "_mask" in img_path.name:
            continue
        subject = img_path.parts[-4]
        eye     = img_path.parts[-3]
        class_samples[f"polyu_{subject}_{eye}_VIS"].append(img_path)

    # UBIRIS
    for img_path in sorted(polar_root.glob("ubiris/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        eye_id = img_path.parts[-2]
        class_samples[f"ubiris_{eye_id}"].append(img_path)

    def subj_key(class_id):
        parts   = class_id.split("_")
        dataset = parts[0]
        if dataset in ("casia", "iitd"):
            return f"{dataset}_{parts[1]}"
        elif dataset == "polyu":
            # NIR y VIS del mismo sujeto van al mismo split
            return f"polyu_{parts[1]}"
        elif dataset == "ubiris":
            return f"ubiris_{int(parts[1]) // 2}"
        return class_id

    splits = _build_splits(class_samples, subj_key, **kwargs)
    _print_stats(splits, ["casia", "iitd", "polyu", "ubiris"],
                 "Joint — CASIA + IITD + PolyU NIR + PolyU VIS + UBIRIS")
    return splits

def _build_splits(class_samples, subject_key_fn,
                  dev_ratio=0.70, enroll_ratio=0.50, seed=42):
    """
    Split 70/30 por sujeto.
    30% evaluación → 50% imágenes enrol / 50% test (mezcladas con seed fija).
    """
    random.seed(seed)

    subject_to_classes = defaultdict(list)
    for class_id in class_samples:
        subject_to_classes[subject_key_fn(class_id)].append(class_id)

    dataset_subjects = defaultdict(list)
    for subject_key, classes in subject_to_classes.items():
        dataset = subject_key.split("_")[0]
        dataset_subjects[dataset].append((subject_key, classes))

    dev_classes  = []
    eval_classes = []
    for dataset, subjects in dataset_subjects.items():
        random.shuffle(subjects)
        n_eval = max(1, round(len(subjects) * (1.0 - dev_ratio)))
        for i, (_, classes) in enumerate(subjects):
            if i < n_eval:
                eval_classes.extend(classes)
            else:
                dev_classes.extend(classes)

    dev_subj = defaultdict(list)
    for class_id in dev_classes:
        dev_subj[subject_key_fn(class_id)].append(class_id)

    dev_list = list(dev_subj.items())
    random.shuffle(dev_list)
    n_val = max(1, round(len(dev_list) * 0.20))

    train_classes = []
    val_classes   = []
    for i, (_, classes) in enumerate(dev_list):
        if i < n_val:
            val_classes.extend(classes)
        else:
            train_classes.extend(classes)

    rng_split      = random.Random(seed)
    enroll_samples = []
    test_samples   = []
    skipped        = 0

    for class_id in eval_classes:
        imgs     = sorted(class_samples[class_id])
        rng_split.shuffle(imgs)
        n_enroll = max(1, round(len(imgs) * enroll_ratio))
        n_test   = len(imgs) - n_enroll
        if n_test < 1:
            enroll_samples.extend([(p, class_id) for p in imgs])
            skipped += 1
            continue
        enroll_samples.extend([(p, class_id) for p in imgs[:n_enroll]])
        test_samples.extend(  [(p, class_id) for p in imgs[n_enroll:]])

    if skipped:
        print(f"  ⚠ {skipped} clases con 1 sola imagen (solo enrolamiento)")

    dev_all_classes  = sorted(set(train_classes + val_classes))
    dev_class_to_idx = {c: i for i, c in enumerate(dev_all_classes)}

    train_samples = [(p, c) for c in train_classes for p in class_samples[c]]
    val_samples   = [(p, c) for c in val_classes   for p in class_samples[c]]
    random.shuffle(train_samples)

    return (train_samples, val_samples, enroll_samples, test_samples,
            dev_class_to_idx)


def build_dataset_phase1(polar_root, **kwargs):
    """Fase 1 — NIR: CASIA + IITD."""
    polar_root    = Path(polar_root)
    class_samples = defaultdict(list)

    for img_path in sorted(polar_root.glob("casia/*/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        subject = img_path.parts[-3]
        eye     = img_path.parts[-2]
        class_samples[f"casia_{subject}_{eye}"].append(img_path)

    for img_path in sorted(polar_root.glob("iitd/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        subject    = img_path.parts[-2]
        parts_name = img_path.stem.replace("_polar", "").split("_")
        eye        = parts_name[1] if len(parts_name) > 1 else "L"
        class_samples[f"iitd_{subject}_{eye}"].append(img_path)

    def subj_key(c):
        p = c.split("_")
        return f"{p[0]}_{p[1]}"

    splits = _build_splits(class_samples, subj_key, **kwargs)
    _print_stats(splits, ["casia", "iitd"], "Fase 1 — NIR (CASIA + IITD)")
    return splits


def build_dataset_phase2(polar_root, **kwargs):
    """Fase 2 — VIS: PolyU VIS + UBIRIS."""
    polar_root    = Path(polar_root)
    class_samples = defaultdict(list)

    for img_path in sorted(polar_root.glob("polyu/*/*/VIS/*polar.png")):
        if "_mask" in img_path.name:
            continue
        subject = img_path.parts[-4]
        eye     = img_path.parts[-3]
        class_samples[f"polyu_{subject}_{eye}_VIS"].append(img_path)

    for img_path in sorted(polar_root.glob("ubiris/*/*.png")):
        if "_mask" in img_path.name or "_polar" not in img_path.name:
            continue
        eye_id = img_path.parts[-2]
        class_samples[f"ubiris_{eye_id}"].append(img_path)

    def subj_key(c):
        parts   = c.split("_")
        dataset = parts[0]
        if dataset == "polyu":
            return f"polyu_{parts[1]}"
        elif dataset == "ubiris":
            return f"ubiris_{int(parts[1]) // 2}"
        return c

    splits = _build_splits(class_samples, subj_key, **kwargs)
    _print_stats(splits, ["polyu", "ubiris"], "Fase 2 — VIS (PolyU VIS + UBIRIS)")
    return splits


def _print_stats(splits, datasets, title):
    train_s, val_s, enroll_s, test_s, dev_c2i = splits

    def count_ds(samples):
        d = defaultdict(int)
        for _, c in samples:
            d[c.split("_")[0]] += 1
        return d

    tr, vl, en, te = (count_ds(s) for s in [train_s, val_s, enroll_s, test_s])

    print(f"\n{'─'*76}")
    print(f"  {title}")
    print(f"{'─'*76}")
    print(f"  {'Dataset':<12} {'Train':>8} {'Val':>8} {'Enrol':>8} {'Test':>8} {'Total':>8}")
    print(f"{'─'*76}")
    for ds in datasets:
        total = tr[ds] + vl[ds] + en[ds] + te[ds]
        print(f"  {ds.upper():<12} {tr[ds]:>8} {vl[ds]:>8} {en[ds]:>8} {te[ds]:>8} {total:>8}")
    print(f"{'─'*76}")
    grand = len(train_s)+len(val_s)+len(enroll_s)+len(test_s)
    print(f"  {'TOTAL':<12} {len(train_s):>8} {len(val_s):>8} "
          f"{len(enroll_s):>8} {len(test_s):>8} {grand:>8}")
    print(f"  Clases dev: {len(dev_c2i)}")
    print(f"{'─'*76}")


# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────

def train_epoch(model, arcface, loader, optimizer, device, scaler,
                epoch, total_epochs, grad_clip=0.0):
    model.train()
    arcface.train()
    total_loss = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"  Época {epoch:3d}/{total_epochs}",
                ncols=90, leave=True)

    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            embeddings = model(imgs)
            loss = arcface(embeddings, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(p for p in model.parameters() if p.requires_grad) +
                list(arcface.parameters()),
                max_norm=grad_clip
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss    = total_loss / (batch_idx + 1)
        elapsed     = time.time() - start_time

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            "t":    f"{elapsed:.0f}s"
        })

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 7. MÉTRICAS
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels     = []
    for imgs, labels in tqdm(loader, desc="  Extrayendo embeddings", leave=False):
        imgs = imgs.to(device)
        emb  = model(imgs)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_embeddings), np.concatenate(all_labels)


def compute_val_metrics(embeddings, labels, n_pairs=20000):
    """EER rápido par a par para monitoreo durante entrenamiento."""
    from sklearn.metrics import roc_curve, auc
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    rng = np.random.default_rng(42)
    idx_by_class = defaultdict(list)
    for i, label in enumerate(labels):
        idx_by_class[int(label)].append(i)

    genuine_scores  = []
    impostor_scores = []
    classes = list(idx_by_class.keys())

    for cls in classes:
        idxs = idx_by_class[cls]
        if len(idxs) < 2:
            continue
        pairs = [(idxs[a], idxs[b])
                 for a in range(len(idxs))
                 for b in range(a+1, len(idxs))]
        if len(pairs) > 10:
            chosen = rng.choice(len(pairs), 10, replace=False)
            pairs  = [pairs[k] for k in chosen]
        for i, j in pairs:
            genuine_scores.append(float(np.dot(embeddings[i], embeddings[j])))

    n_imp = min(n_pairs, len(genuine_scores) * 5)
    for _ in range(n_imp):
        c1, c2 = rng.choice(len(classes), 2, replace=False)
        i = rng.choice(idx_by_class[classes[c1]])
        j = rng.choice(idx_by_class[classes[c2]])
        impostor_scores.append(float(np.dot(embeddings[i], embeddings[j])))

    scores     = genuine_scores + impostor_scores
    labels_bin = [1]*len(genuine_scores) + [0]*len(impostor_scores)

    fpr, tpr, _ = roc_curve(labels_bin, scores)
    roc_auc     = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100

    def tar_at_far(target_far):
        return tpr[np.argmin(np.abs(fpr - target_far))] * 100

    mu_g   = np.mean(genuine_scores)
    mu_i   = np.mean(impostor_scores)
    std_g  = np.std(genuine_scores)
    std_i  = np.std(impostor_scores)
    dprime = abs(mu_g - mu_i) / np.sqrt(0.5 * (std_g**2 + std_i**2))

    return {
        "eer":             round(eer, 4),
        "tar_at_far01":    round(tar_at_far(0.001), 4),
        "tar_at_far1":     round(tar_at_far(0.01),  4),
        "auc":             round(roc_auc * 100, 4),
        "dprime":          round(float(dprime), 4),
        "mean_genuine":    round(float(mu_g), 4),
        "mean_impostor":   round(float(mu_i), 4),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs":len(impostor_scores),
        "fpr":             fpr.tolist(),
        "tpr":             tpr.tolist(),
        "genuine_scores":  genuine_scores[:2000],
        "impostor_scores": impostor_scores[:2000],
    }


@torch.no_grad()
def compute_final_eer(model, enroll_samples, test_samples, device, batch_size=64):
    """Evaluación final exhaustiva con score máximo."""
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    all_eval_classes  = sorted(set(c for _, c in enroll_samples + test_samples))
    eval_class_to_idx = {c: i for i, c in enumerate(all_eval_classes)}

    print("\n  Extrayendo embeddings de enrolamiento...")
    enroll_ds     = IrisDataset(enroll_samples, eval_class_to_idx, None)
    enroll_loader = DataLoader(enroll_ds, batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    enroll_embs, enroll_labels = extract_embeddings(model, enroll_loader, device)

    print("  Extrayendo embeddings de test...")
    test_ds     = IrisDataset(test_samples, eval_class_to_idx, None)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    test_embs, test_labels = extract_embeddings(model, test_loader, device)

    enroll_by_class = defaultdict(list)
    for emb, label in zip(enroll_embs, enroll_labels):
        enroll_by_class[int(label)].append(emb)

    enroll_arrays    = {cls: np.array(embs) for cls, embs in enroll_by_class.items()}
    enrolled_classes = sorted(enroll_arrays.keys())

    genuine_scores  = []
    impostor_scores = []

    for test_emb, test_label in zip(test_embs, test_labels):
        test_label = int(test_label)
        if test_label in enroll_arrays:
            sims = enroll_arrays[test_label] @ test_emb
            genuine_scores.append(float(np.max(sims)))
        for imp_cls in enrolled_classes:
            if imp_cls == test_label:
                continue
            sims = enroll_arrays[imp_cls] @ test_emb
            impostor_scores.append(float(np.max(sims)))

    scores     = genuine_scores + impostor_scores
    labels_bin = [1]*len(genuine_scores) + [0]*len(impostor_scores)

    fpr, tpr, _ = roc_curve(labels_bin, scores)
    roc_auc     = sklearn_auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100

    def tar_at_far(target_far):
        return tpr[np.argmin(np.abs(fpr - target_far))] * 100

    mu_g   = np.mean(genuine_scores)
    mu_i   = np.mean(impostor_scores)
    std_g  = np.std(genuine_scores)
    std_i  = np.std(impostor_scores)
    dprime = abs(mu_g - mu_i) / np.sqrt(0.5 * (std_g**2 + std_i**2))

    return {
        "eer":          round(eer, 4),
        "tar_at_far01": round(tar_at_far(0.001), 4),
        "tar_at_far1":  round(tar_at_far(0.01),  4),
        "auc":          round(roc_auc * 100, 4),
        "dprime":       round(float(dprime), 4),
        "mean_genuine": round(float(mu_g), 4),
        "mean_impostor":round(float(mu_i), 4),
        "n_genuine":    len(genuine_scores),
        "n_impostor":   len(impostor_scores),
        "fpr":          fpr.tolist(),
        "tpr":          tpr.tolist(),
        "genuine_scores":  genuine_scores[:2000],
        "impostor_scores": impostor_scores[:2000],
    }


# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_training(log, results_dir, phase):
    epochs = [e["epoch"] for e in log]
    losses = [e["loss"]  for e in log]
    lrs    = [e["lr"]    for e in log]
    eval_log    = [e for e in log if "eer" in e]
    eval_epochs = [e["epoch"] for e in eval_log]
    eers        = [e["eer"]   for e in eval_log]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Entrenamiento {phase.upper()} — EfficientNetV2-S + ArcFace",
                 fontsize=12)

    axes[0].plot(epochs, losses, 'b-o', markersize=3)
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].set_title("ArcFace Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_epochs, eers, 'r-o', markersize=4)
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("EER (%)")
    axes[1].set_title("EER val par a par (↓ mejor)")
    axes[1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1%')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, lrs, 'g-', linewidth=2)
    axes[2].set_xlabel("Época"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = str(Path(results_dir) / f"training_curves_{phase}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  ✓ Curvas guardadas en {out}")


def plot_score_distribution(metrics, results_dir, tag):
    gen = metrics["genuine_scores"]
    imp = metrics["impostor_scores"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Scores — {tag} | EER={metrics['eer']:.2f}%", fontsize=12)

    ax1.hist(imp, bins=60, alpha=0.6, color='red',
             label=f"Impostores (μ={metrics['mean_impostor']:.3f})")
    ax1.hist(gen, bins=60, alpha=0.6, color='blue',
             label=f"Genuinos (μ={metrics['mean_genuine']:.3f})")
    ax1.set_xlabel("Similitud coseno"); ax1.set_ylabel("Frecuencia")
    ax1.set_title(f"d-prime = {metrics['dprime']:.3f}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    fpr = np.array(metrics["fpr"])
    tpr = np.array(metrics["tpr"])
    ax2.plot(fpr*100, tpr*100, 'b-', linewidth=2,
             label=f"AUC={metrics['auc']:.2f}%")
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax2.set_xlabel("FAR (%)"); ax2.set_ylabel("TAR (%)")
    ax2.set_title(f"ROC | TAR@FAR=1%: {metrics['tar_at_far1']:.2f}%")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = str(Path(results_dir) / f"score_dist_{tag}.png")
    plt.savefig(out, dpi=150); plt.close()


# ─────────────────────────────────────────────
# 9. GUARDAR EMBEDDINGS
# ─────────────────────────────────────────────

def save_embeddings(model, samples, class_to_idx, features_root,
                    split_name, device, batch_size=64):
    dataset = IrisDataset(samples, class_to_idx, None)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)
    embeddings, labels = extract_embeddings(model, loader, device)

    out_dir = Path(features_root) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "embeddings.npy"), embeddings)
    np.save(str(out_dir / "labels.npy"),     labels)

    class_strings = [c for _, c in samples]
    paths         = [str(p) for p, _ in samples]
    with open(str(out_dir / "class_strings.json"), "w") as f:
        json.dump(class_strings, f)
    with open(str(out_dir / "paths.json"), "w") as f:
        json.dump(paths, f)

    print(f"  ✓ {split_name:<14}: {embeddings.shape} → {out_dir}")
    return embeddings, labels


# ─────────────────────────────────────────────
# 10. FLUJO GENÉRICO DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def run_training(model, arcface, train_samples, val_samples,
                 enroll_samples, test_samples, dev_class_to_idx,
                 args, device, results_dir, features_root,
                 phase, model_path, epochs, lr, warmup, margin):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.02)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), value=0),
    ])

    train_dataset = IrisDataset(train_samples, dev_class_to_idx, train_transform)
    val_dataset   = IrisDataset(val_samples,   dev_class_to_idx, None)

    class_counts = defaultdict(int)
    for _, c in train_samples:
        class_counts[dev_class_to_idx[c]] += 1
    weights = [1.0 / class_counts[dev_class_to_idx[c]] for _, c in train_samples]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False,    num_workers=4, pin_memory=True)

    # Solo parámetros entrenables (respeta freeze_backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad] +                        list(arcface.parameters())
    # lr inicial: si hay warmup arranca en lr/warmup_epochs (primer paso del warmup)
    # si no hay warmup arranca directamente en lr
    lr_init = lr if warmup == 0 else lr / max(warmup, 1)
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr_init,
        weight_decay=1e-4
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup, total_epochs=epochs,
        lr_max=lr, lr_min=1e-6
    )
    scaler           = torch.amp.GradScaler('cuda')
    best_eer         = 100.0
    train_log        = []
    epochs_no_improve = 0   # contador para early stopping

    print(f"\n🏋️  [{phase}] {epochs} épocas | lr={lr} | "
          f"margin={margin} | warmup={warmup} | "
          f"eval_every={args.eval_every} | "
          f"early_stopping={args.early_stopping}\n")

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, arcface, train_loader,
                           optimizer, device, scaler,
                           epoch, epochs, args.grad_clip)
        lr_cur = scheduler.step()
        log_entry = {"epoch": epoch, "loss": round(loss, 6), "lr": lr_cur}

        if epoch % args.eval_every == 0 or epoch == epochs:
            val_embs, val_labels = extract_embeddings(model, val_loader, device)
            metrics = compute_val_metrics(val_embs, val_labels)
            log_entry.update({k: metrics[k] for k in
                              ["eer","tar_at_far01","tar_at_far1","auc","dprime"]})

            print(f"\n  📊 [{phase}] Val época {epoch}:")
            print(f"     EER          : {metrics['eer']:.2f}%")
            print(f"     TAR@FAR=0.1% : {metrics['tar_at_far01']:.2f}%")
            print(f"     TAR@FAR=1%   : {metrics['tar_at_far1']:.2f}%")
            print(f"     AUC          : {metrics['auc']:.2f}%")
            print(f"     d-prime      : {metrics['dprime']:.3f}")
            print(f"     Score gen.   : {metrics['mean_genuine']:.3f}")
            print(f"     Score imp.   : {metrics['mean_impostor']:.3f}")

            plot_score_distribution(metrics, results_dir,
                                    f"val_{phase}_epoch{epoch:03d}")

            with open(str(Path(results_dir) /
                      f"metrics_val_{phase}_epoch{epoch:03d}.json"), "w") as f:
                json.dump({k: v for k, v in metrics.items()
                           if k not in ["fpr","tpr","genuine_scores","impostor_scores"]},
                          f, indent=2)

            if metrics["eer"] < best_eer:
                best_eer          = metrics["eer"]
                epochs_no_improve = 0
                torch.save(model.state_dict(), str(model_path))
                print(f"\n  ✓ Mejor modelo guardado (EER val={best_eer:.2f}%)")
            else:
                epochs_no_improve += args.eval_every
                print(f"  Sin mejora ({epochs_no_improve} épocas, "
                      f"límite={args.early_stopping})")
                if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
                    print(f"\n  🛑 Early stopping en época {epoch}")
                    train_log.append(log_entry)
                    break
        else:
            print(f"  Época {epoch:3d}/{epochs} | "
                  f"Loss: {loss:.4f} | lr: {lr_cur:.2e}")

        train_log.append(log_entry)

    total_time = time.time() - total_start
    print(f"\n{'═'*52}")
    print(f"  {phase.upper()} COMPLETADO")
    print(f"{'═'*52}")
    print(f"  Tiempo       : {total_time/60:.1f} min")
    print(f"  Mejor EER val: {best_eer:.2f}%")
    print(f"{'═'*52}\n")

    with open(str(Path(results_dir) / f"training_log_{phase}.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    if any("eer" in e for e in train_log):
        plot_training(train_log, results_dir, phase)

    # Cargar mejor modelo
    model.load_state_dict(torch.load(str(model_path),
                                     map_location=device, weights_only=True))

    # Evaluación final
    print(f"\n🎯 [{phase}] Evaluación final...")
    final_metrics = compute_final_eer(model, enroll_samples, test_samples,
                                      device, args.batch_size)

    print(f"\n{'═'*52}")
    print(f"  EVALUACIÓN FINAL — {phase.upper()}")
    print(f"{'═'*52}")
    print(f"  EER            : {final_metrics['eer']:.2f}%")
    print(f"  TAR@FAR=0.1%   : {final_metrics['tar_at_far01']:.2f}%")
    print(f"  TAR@FAR=1%     : {final_metrics['tar_at_far1']:.2f}%")
    print(f"  AUC            : {final_metrics['auc']:.2f}%")
    print(f"  d-prime        : {final_metrics['dprime']:.3f}")
    print(f"  Score genuino  : {final_metrics['mean_genuine']:.3f}")
    print(f"  Score impostor : {final_metrics['mean_impostor']:.3f}")
    print(f"  Pares genuinos : {final_metrics['n_genuine']}")
    print(f"  Pares impostores: {final_metrics['n_impostor']}")
    print(f"{'═'*52}\n")

    with open(str(Path(results_dir) / f"metrics_final_{phase}.json"), "w") as f:
        json.dump({k: v for k, v in final_metrics.items()
                   if k not in ["fpr","tpr","genuine_scores","impostor_scores"]},
                  f, indent=2)
    plot_score_distribution(final_metrics, results_dir, f"final_{phase}")

    # Guardar embeddings
    print(f"\n💾 [{phase}] Guardando embeddings...")
    all_eval_classes  = sorted(set(c for _, c in enroll_samples + test_samples))
    eval_class_to_idx = {c: i for i, c in enumerate(all_eval_classes)}

    save_embeddings(model, train_samples,  dev_class_to_idx,
                    features_root, f"train_{phase}",  device, args.batch_size)
    save_embeddings(model, val_samples,    dev_class_to_idx,
                    features_root, f"val_{phase}",    device, args.batch_size)
    save_embeddings(model, enroll_samples, eval_class_to_idx,
                    features_root, f"enroll_{phase}", device, args.batch_size)
    save_embeddings(model, test_samples,   eval_class_to_idx,
                    features_root, f"test_{phase}",   device, args.batch_size)


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        "EfficientNetV2-S + ArcFace — Transfer learning NIR→VIS")

    parser.add_argument("--run_name",  type=str, default="exp06")
    parser.add_argument("--mode",      type=str, default="phase1",
                        choices=["phase1", "phase2", "joint", "extract"],
                        help="phase1: NIR | phase2: VIS fine-tuning | "
                             "joint: todos los datasets juntos | "
                             "extract: solo embeddings")
    parser.add_argument("--phase",     type=str, default="phase2",
                        help="Fase a usar en modo extract (phase1 o phase2)")

    # Comunes
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--emb_dim",         type=int,   default=1024)
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--grad_clip",       type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--eval_every",      type=int,   default=5)
    parser.add_argument("--enroll_ratio",    type=float, default=0.5)
    parser.add_argument("--checkpoint_every", type=int,   default=20,
                        help="Guardar checkpoint cada N épocas (0=desactivado)")
    parser.add_argument("--early_stopping",  type=int,   default=15,
                        help="Épocas sin mejora antes de parar (0=desactivado)")

    # Fase 1
    parser.add_argument("--epochs_phase1",  type=int,   default=50)
    parser.add_argument("--lr_phase1",      type=float, default=1e-4)
    parser.add_argument("--margin_phase1",  type=float, default=0.4)
    parser.add_argument("--warmup_phase1",  type=int,   default=5)

    # Fase 2
    parser.add_argument("--epochs_phase2",  type=int,   default=30)
    parser.add_argument("--lr_phase2",      type=float, default=1e-4)
    parser.add_argument("--margin_phase2",  type=float, default=0.3)
    parser.add_argument("--warmup_phase2",  type=int,   default=5)
    parser.add_argument("--unfreeze_blocks", type=int,  default=2,
                        help="Últimos N bloques del backbone a descongelar en fase 2 (0=todos congelados)")

    args = parser.parse_args()

    config = load_config("config.yaml")
    set_seed(42)

    polar_root    = config["paths"]["data"]["polar"]
    results_dir   = str(Path(config["paths"]["results"])               / args.run_name)
    models_dir    = str(Path(config["paths"]["models"]["efficientnet"]) / args.run_name)
    features_root = str(Path(config["paths"]["data"]["features"])      / args.run_name)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(features_root).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Experimento : {args.run_name} | Modo: {args.mode}")
    print(f"   Dispositivo : {device}")
    if device.type == "cuda":
        print(f"   GPU         : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model_path_phase1 = Path(models_dir) / "phase1.pth"
    model_path_phase2 = Path(models_dir) / "phase2.pth"

    # ── FASE 1 ────────────────────────────────────────────────────────────
    if args.mode == "phase1":
        print("\n📂 Dataset fase 1 (CASIA + IITD — NIR)...")
        (train_s, val_s, enroll_s, test_s,
         dev_c2i) = build_dataset_phase1(polar_root,
                                          enroll_ratio=args.enroll_ratio)

        print(f"\n🧠 Modelo fase 1 desde ImageNet pretrained...")
        model   = IrisEfficientNet(embedding_dim=args.emb_dim,
                                   pretrained=True,
                                   dropout=args.dropout).to(device)
        arcface = ArcFaceLoss(args.emb_dim, len(dev_c2i),
                              margin=args.margin_phase1,
                              label_smoothing=args.label_smoothing).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parámetros: {n_params/1e6:.1f}M | Clases NIR: {len(dev_c2i)}")

        run_training(model, arcface, train_s, val_s, enroll_s, test_s,
                     dev_c2i, args, device, results_dir, features_root,
                     "phase1", model_path_phase1,
                     args.epochs_phase1, args.lr_phase1,
                     args.warmup_phase1, args.margin_phase1)

    # ── FASE 2 ────────────────────────────────────────────────────────────
    elif args.mode == "phase2":
        if not model_path_phase1.exists():
            print(f"  ⚠ Modelo fase 1 no encontrado en {model_path_phase1}")
            print(f"    Ejecuta primero: --mode phase1")
            return

        print("\n📂 Dataset fase 2 (PolyU VIS + UBIRIS)...")
        (train_s, val_s, enroll_s, test_s,
         dev_c2i) = build_dataset_phase2(polar_root,
                                          enroll_ratio=args.enroll_ratio)

        print(f"\n🧠 Cargando fase 1 — fine-tuning completo con lr={args.lr_phase2:.0e}...")
        model = IrisEfficientNet(embedding_dim=args.emb_dim,
                                 pretrained=False,
                                 dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(str(model_path_phase1),
                                         map_location=device, weights_only=True))
        model.freeze_backbone(unfreeze_last_n_blocks=args.unfreeze_blocks)

        arcface = ArcFaceLoss(args.emb_dim, len(dev_c2i),
                              margin=args.margin_phase2,
                              label_smoothing=args.label_smoothing).to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parámetros entrenables: {trainable/1e6:.1f}M | Clases VIS: {len(dev_c2i)}")

        run_training(model, arcface, train_s, val_s, enroll_s, test_s,
                     dev_c2i, args, device, results_dir, features_root,
                     "phase2", model_path_phase2,
                     args.epochs_phase2, args.lr_phase2,
                     args.warmup_phase2, args.margin_phase2)

    # ── JOINT ─────────────────────────────────────────────────────────────
    elif args.mode == "joint":
        print("\n📂 Dataset conjunto (CASIA + IITD + PolyU NIR+VIS + UBIRIS)...")
        (train_s, val_s, enroll_s, test_s,
         dev_c2i) = build_dataset_joint(polar_root,
                                         enroll_ratio=args.enroll_ratio)

        model_path_joint = Path(models_dir) / "joint.pth"

        print(f"\n🧠 Inicializando modelo (entrenamiento conjunto)...")
        print(f"   emb_dim={args.emb_dim} | margin={args.margin_phase1} | "
              f"dropout={args.dropout} | warmup={args.warmup_phase1} | "
              f"epochs={args.epochs_phase1}")

        model   = IrisEfficientNet(embedding_dim=args.emb_dim,
                                   pretrained=True,
                                   dropout=args.dropout).to(device)
        arcface = ArcFaceLoss(args.emb_dim, len(dev_c2i),
                              margin=args.margin_phase1,
                              label_smoothing=args.label_smoothing).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parámetros: {n_params/1e6:.1f}M | Clases: {len(dev_c2i)}")

        run_training(model, arcface, train_s, val_s, enroll_s, test_s,
                     dev_c2i, args, device, results_dir, features_root,
                     "joint", model_path_joint,
                     args.epochs_phase1, args.lr_phase1,
                     args.warmup_phase1, args.margin_phase1)

    # ── EXTRACT ───────────────────────────────────────────────────────────
    elif args.mode == "extract":
        phase      = args.phase
        model_path = model_path_phase1 if phase == "phase1" else model_path_phase2
        build_fn   = build_dataset_phase1 if phase == "phase1" else build_dataset_phase2

        if not model_path.exists():
            print(f"  ⚠ Modelo no encontrado en {model_path}")
            return

        print(f"\n📂 Dataset {phase}...")
        (train_s, val_s, enroll_s, test_s,
         dev_c2i) = build_fn(polar_root, enroll_ratio=args.enroll_ratio)

        model = IrisEfficientNet(embedding_dim=args.emb_dim,
                                 pretrained=False,
                                 dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(str(model_path),
                                         map_location=device, weights_only=True))
        print(f"  ✓ Modelo {phase} cargado")

        print(f"\n🎯 Evaluación final ({phase})...")
        final_metrics = compute_final_eer(model, enroll_s, test_s,
                                          device, args.batch_size)
        print(f"  EER: {final_metrics['eer']:.2f}%")

        all_eval_classes  = sorted(set(c for _, c in enroll_s + test_s))
        eval_class_to_idx = {c: i for i, c in enumerate(all_eval_classes)}
        print(f"\n💾 Guardando embeddings {phase}...")
        save_embeddings(model, train_s,  dev_c2i,
                        features_root, f"train_{phase}",  device, args.batch_size)
        save_embeddings(model, val_s,    dev_c2i,
                        features_root, f"val_{phase}",    device, args.batch_size)
        save_embeddings(model, enroll_s, eval_class_to_idx,
                        features_root, f"enroll_{phase}", device, args.batch_size)
        save_embeddings(model, test_s,   eval_class_to_idx,
                        features_root, f"test_{phase}",   device, args.batch_size)

    print("\n✅ Completado.\n")


if __name__ == "__main__":
    main()
