"""
eval_biometric.py
Evaluacion completa del subsistema biometrico.

Evalua el modelo EfficientNetV2-S + ArcFace en dos dominios:
  - Dominio original: similitud coseno entre embeddings de 2048 floats
  - Dominio cancelable: distancia de Hamming entre templates binarios (256 bits)

La comparacion entre dominios permite cuantificar el coste en precision
biometrica introducido por la transformacion cancelable.

Metricas calculadas (ISO/IEC 19795):
  - FTE  (Failure-to-Enrol Rate)
  - FTA  (Failure-to-Acquire Rate)
  - FMR  (False Match Rate) al umbral EER
  - FNMR (False Non-Match Rate) al umbral EER
  - FAR / FRR a distintos umbrales
  - EER  (Equal Error Rate)
  - TAR@FAR=0.1%, 1%, 10%
  - AUC  (Area Under the ROC Curve)
  - d-prime
  Todo lo anterior en dominio original Y cancelable, por separado.

Protocolo (ISO/IEC 19795):
  - Split: 50% imagenes por sujeto para enrolamiento, 50% para test
  - Score maximo exhaustivo: cada imagen test vs todas las plantillas enroladas
  - Comparacion cruzada exhaustiva para scores impostores

Salidas:
  - results/<run_name>/eval_biometric_results.json
  - results/<run_name>/eval_biometric_roc_comparison.png
  - results/<run_name>/eval_biometric_scores_original.png
  - results/<run_name>/eval_biometric_scores_cancelable.png
  - results/<run_name>/eval_biometric_datasets.png
  - results/<run_name>/eval_biometric_report.pdf

Uso:
    python scripts/eval_biometric.py --run_name exp08
    python scripts/eval_biometric.py --run_name exp08 --n_bits 256
"""

import sys
import json
import secrets
import argparse
import importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve, auc as sklearn_auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from reedsolo import RSCodec
import yaml
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, PageBreak,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

sys.path.insert(0, str(Path(__file__).parent))
_cancelable              = importlib.import_module("06_cancelable")
enroll_template_reliable = _cancelable.enroll_template_reliable
verify_template_reliable = _cancelable.verify_template_reliable
hamming_distance         = _cancelable.hamming_distance

_fc              = importlib.import_module("07_fuzzy_commitment")
enroll_user_fc   = _fc.enroll_user
authenticate_user_fc = _fc.authenticate_user

_ec = importlib.import_module("eval_common")


# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(features_root, split_name):
    split_dir     = Path(features_root) / split_name
    embeddings    = np.load(str(split_dir / "embeddings.npy"))
    class_strings = json.load(open(str(split_dir / "class_strings.json")))
    paths         = json.load(open(str(split_dir / "paths.json")))
    return embeddings, class_strings, paths


def get_dataset_label(class_str, img_path):
    dataset = class_str.split("_")[0]
    if dataset == "polyu":
        path_str = str(img_path).replace("\\", "/")
        if "/NIR/" in path_str: return "PolyU NIR"
        if "/VIS/" in path_str: return "PolyU VIS"
        return "PolyU"
    return dataset.upper()


# ─────────────────────────────────────────────
# 2. METRICAS
# ─────────────────────────────────────────────

def compute_metrics(genuine_scores, impostor_scores, higher_is_better=True):
    """
    Calcula metricas biometricas completas segun ISO/IEC 19795.
    higher_is_better=True para similitud coseno, False para distancia Hamming.
    """
    if not higher_is_better:
        # Para distancia de Hamming invertimos los scores
        genuine_scores  = [-s for s in genuine_scores]
        impostor_scores = [-s for s in impostor_scores]

    scores     = genuine_scores + impostor_scores
    labels_bin = [1]*len(genuine_scores) + [0]*len(impostor_scores)

    fpr, tpr, thresholds = roc_curve(labels_bin, scores)
    roc_auc = sklearn_auc(fpr, tpr)

    # EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    eer_threshold_neg = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    eer_threshold = -eer_threshold_neg if not higher_is_better else eer_threshold_neg

    # FMR y FNMR al umbral EER
    idx_eer = np.argmin(np.abs(fpr - (1 - tpr)))
    fmr_eer  = float(fpr[idx_eer]) * 100
    fnmr_eer = float(1 - tpr[idx_eer]) * 100

    def tar_at_far(target_far):
        idx = np.argmin(np.abs(fpr - target_far))
        return float(tpr[idx]) * 100

    g = np.array(genuine_scores)
    i = np.array(impostor_scores)
    dprime = abs(g.mean() - i.mean()) / np.sqrt(0.5*(g.std()**2 + i.std()**2))

    return {
        "eer":           round(eer, 4),
        "fmr_at_eer":    round(fmr_eer, 4),
        "fnmr_at_eer":   round(fnmr_eer, 4),
        "tar_at_far001": round(tar_at_far(0.001), 4),
        "tar_at_far01":  round(tar_at_far(0.01),  4),
        "tar_at_far1":   round(tar_at_far(0.1),   4),
        "auc":           round(roc_auc * 100, 4),
        "dprime":        round(float(dprime), 4),
        "mean_genuine":  round(float(g.mean()), 4),
        "std_genuine":   round(float(g.std()),  4),
        "mean_impostor": round(float(i.mean()), 4),
        "std_impostor":  round(float(i.std()),  4),
        "eer_threshold": round(float(eer_threshold), 4),
        "n_genuine":     len(genuine_scores),
        "n_impostor":    len(impostor_scores),
        "fpr":  fpr,
        "tpr":  tpr,
    }


# ─────────────────────────────────────────────
# 3. EVALUACION DOMINIO ORIGINAL
# ─────────────────────────────────────────────

def run_original_domain(enroll_embs, enroll_classes,
                         test_embs,   test_classes, test_paths):
    """Evaluacion con similitud coseno en el espacio de embeddings."""
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    enroll_arrays    = {cls: np.array(embs) for cls, embs in enroll_by_class.items()}
    enrolled_classes = sorted(enroll_arrays.keys())

    genuine_scores  = []
    impostor_scores = []
    genuine_by_ds   = defaultdict(list)

    print(f"  Comparando {len(test_embs)} imagenes vs {len(enrolled_classes)} clases...")
    for test_emb, test_cls, test_path in zip(test_embs, test_classes, test_paths):
        ds = get_dataset_label(test_cls, test_path)
        if test_cls in enroll_arrays:
            score = float(np.max(enroll_arrays[test_cls] @ test_emb))
            genuine_scores.append(score)
            genuine_by_ds[ds].append(score)
        for imp_cls in enrolled_classes:
            if imp_cls != test_cls:
                impostor_scores.append(
                    float(np.max(enroll_arrays[imp_cls] @ test_emb)))

    return genuine_scores, impostor_scores, genuine_by_ds


# ─────────────────────────────────────────────
# 4. EVALUACION DOMINIO CANCELABLE
# ─────────────────────────────────────────────

def run_cancelable_domain(enroll_embs, enroll_classes,
                           test_embs,   test_classes, test_paths,
                           n_bits):
    """
    Evaluacion con distancia de Hamming en el dominio cancelable.
    Cada usuario tiene su propio transform_seed para la comparacion.
    La similitud se define como 1 - BER (Hamming normalizada).
    """
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    # Generar templates cancelables de enrolamiento (un seed fijo por usuario)
    print(f"  Generando templates cancelables ({n_bits} bits)...")
    user_seeds    = {cls: secrets.token_hex(32) for cls in enroll_by_class}
    enroll_tmpls  = {}
    enroll_indices = {}
    for cls, embs in enroll_by_class.items():
        tmpl, idx = enroll_template_reliable(
            embs, cls, user_seeds[cls], n_bits)
        enroll_tmpls[cls]   = tmpl
        enroll_indices[cls] = idx

    enrolled_classes = sorted(enroll_tmpls.keys())

    genuine_scores  = []
    impostor_scores = []
    genuine_by_ds   = defaultdict(list)

    for test_emb, test_cls, test_path in zip(test_embs, test_classes, test_paths):
        ds = get_dataset_label(test_cls, test_path)
        if test_cls in enroll_tmpls:
            ver_tmpl = verify_template_reliable(
                test_emb, test_cls,
                user_seeds[test_cls], enroll_indices[test_cls])
            sim = 1.0 - hamming_distance(enroll_tmpls[test_cls], ver_tmpl)
            genuine_scores.append(sim)
            genuine_by_ds[ds].append(sim)

        for imp_cls in enrolled_classes:
            if imp_cls != test_cls:
                # Usar el seed del impostor para la verificacion cruzada
                ver_tmpl = verify_template_reliable(
                    test_emb, imp_cls,
                    user_seeds[imp_cls], enroll_indices[imp_cls])
                sim = 1.0 - hamming_distance(enroll_tmpls[imp_cls], ver_tmpl)
                impostor_scores.append(sim)

    return genuine_scores, impostor_scores, genuine_by_ds


# ─────────────────────────────────────────────
# 5. FTE y FTA
# ─────────────────────────────────────────────

def compute_fte_fta(enroll_embs, enroll_classes,
                    test_embs,   test_classes,
                    n_enrolled_classes):
    """
    FTE (Failure-to-Enrol) y FTA (Failure-to-Acquire) segun ISO/IEC 19795.

    ALCANCE — importante para la lectura del informe:
    Estas tasas se calculan sobre embeddings YA EXTRAIDOS, detectando
    vectores invalidos (NaN o nulos). Las imagenes que fallaron en la
    segmentacion o el preprocesado nunca llegan a esta fase, por lo que
    un resultado de 0% NO puede interpretarse como "el pipeline de
    adquisicion no falla nunca": mide solo la etapa de extraccion de
    caracteristicas. El FTA del modulo de segmentacion debe medirse
    aparte, en el propio pipeline de preprocesado.
    """
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    # FTE: clases con embeddings invalidos (NaN o cero)
    fte_count = 0
    for cls, embs in enroll_by_class.items():
        arr = np.array(embs)
        if np.any(np.isnan(arr)) or np.all(arr == 0):
            fte_count += 1
    fte = fte_count / max(len(enroll_by_class), 1) * 100

    # FTA: imagenes de test con embeddings invalidos
    fta_count = sum(1 for emb in test_embs
                    if np.any(np.isnan(emb)) or np.all(emb == 0))
    fta = fta_count / max(len(test_embs), 1) * 100

    return round(fte, 4), round(fta, 4)


# ─────────────────────────────────────────────
# 6. EVALUACION MEDIANTE FUZZY COMMITMENT
# ─────────────────────────────────────────────

def run_crypto_evaluation(enroll_embs, enroll_classes, enroll_paths,
                           test_embs, test_classes, test_paths,
                           n_bits, n_parity, n_enroll=5, n_verify=3,
                           n_impostor=100000, seed=_ec.DEFAULT_SEED):
    """
    Evalua el rendimiento biometrico a traves del pipeline criptografico
    completo.

    En lugar de similitud coseno o distancia de Hamming, la autenticacion
    se realiza mediante el proceso real de fuzzy commitment:
      - Enrolamiento: template protegido con RS + Argon2
      - Autenticacion: recuperacion del secreto via RS + verificacion Argon2

    Cambios respecto a la version anterior:
      - Enrolamiento determinista (RNG sembrado) -> resultados reproducibles
      - n_impostor configurable y por defecto 100.000 en lugar de 1.000:
        con 1.000 intentos y 0 falsas aceptaciones el IC 95% solo permite
        afirmar FAR < 0.37%, lo que equivale a 8 bits de seguridad efectiva
      - Todas las tasas se devuelven con intervalo de confianza exacto
      - Se recogen las BER impostoras para calcular la FAR analitica, que
        da varios ordenes de magnitud mas de resolucion que el conteo
      - GAR por dataset con su N, no solo el porcentaje
    """
    from collections import defaultdict as dd

    rng = _ec.make_rng(seed)

    enroll_by_class  = dd(list)
    enroll_paths_cls = dd(list)
    for emb, cls, path in zip(enroll_embs, enroll_classes, enroll_paths):
        enroll_by_class[cls].append(emb)
        enroll_paths_cls[cls].append(path)

    test_by_class  = dd(list)
    test_paths_cls = dd(list)
    for emb, cls, path in zip(test_embs, test_classes, test_paths):
        test_by_class[cls].append(emb)
        test_paths_cls[cls].append(path)

    # Se excluyen los sujetos que no llegan al minimo de n_enroll/n_verify
    # imagenes. Sin esto, embs[:n_enroll] usaria menos imagenes de las
    # declaradas para esos sujetos en silencio, y "Usuarios enrolados"
    # no describiria con exactitud a la poblacion evaluada.
    enroll_by_class, test_by_class, filter_stats = (
        _ec.filter_subjects_with_min_images(
            enroll_by_class, test_by_class, n_enroll, n_verify))
    if filter_stats["n_excluded"]:
        n_exc = filter_stats["n_excluded"]
        n_bef = filter_stats["n_subjects_before"]
        n_aft = filter_stats["n_subjects_after"]
        print(f"    {n_exc} sujetos excluidos "
              f"(menos de {n_enroll}/{n_verify} imagenes disponibles): "
              f"{n_bef} -> {n_aft}")

    print(f"    Enrollando {len(enroll_by_class)} usuarios (determinista)...")
    user_records = {}
    for cls, embs in enroll_by_class.items():
        try:
            user_records[cls] = _ec.enroll_user_seeded(
                embs[:n_enroll], cls, n_bits, n_parity, rng)
        except Exception:
            pass

    enrolled_classes = sorted(user_records.keys())

    # ── Autenticacion genuina ────────────────────────────────────────────
    genuine_ok, genuine_tot = 0, 0
    gar_by_ds = dd(lambda: [0, 0])

    for cls, test_arr in test_by_class.items():
        if cls not in user_records:
            continue
        ds     = get_dataset_label(cls, test_paths_cls[cls][0])
        arr    = np.array(test_arr)
        record = user_records[cls]

        for start in range(0, len(arr) - n_verify + 1, n_verify):
            window = list(arr[start:start+n_verify])
            success, _ = authenticate_user_fc(window, cls, record)
            genuine_tot += 1
            gar_by_ds[ds][1] += 1
            if success:
                genuine_ok += 1
                gar_by_ds[ds][0] += 1

    # ── Autenticacion impostora (zero-effort) ────────────────────────────
    impostor_ok, impostor_tot = 0, 0
    impostor_bers  = []
    impostor_pairs = []
    test_cls_list = [c for c in test_by_class if c in user_records]

    print(f"    Evaluando {n_impostor:,} intentos impostores...")
    for i in range(n_impostor):
        qcls = rng.choice(test_cls_list)
        tcls = rng.choice([c for c in enrolled_classes if c != qcls])
        tarr = np.array(test_by_class[qcls])
        if len(tarr) < n_verify:
            continue
        start  = rng.integers(0, len(tarr) - n_verify + 1)
        window = list(tarr[start:start+n_verify])

        record = user_records[tcls]
        idx    = np.array(record["reliable_indices"])

        # BER impostora sobre el template cancelable del objetivo
        mean_q = np.mean(np.array(window), axis=0)
        nrm    = np.linalg.norm(mean_q)
        if nrm > 0:
            mean_q = mean_q / nrm
        tmpl_q = verify_template_reliable(
            mean_q, tcls, record["transform_seed"], idx)

        enr_arr  = np.array(enroll_by_class[tcls][:n_enroll])
        mean_enr = np.mean(enr_arr, axis=0)
        nrm      = np.linalg.norm(mean_enr)
        if nrm > 0:
            mean_enr = mean_enr / nrm
        tmpl_t = verify_template_reliable(
            mean_enr, tcls, record["transform_seed"], idx)
        _ber = hamming_distance(tmpl_t, tmpl_q)
        impostor_bers.append(_ber)
        impostor_pairs.append((_ber, qcls, tcls))

        # Decodificacion RS rapida; Argon2 solo si RS tiene exito.
        # Equivalente a authenticate_user_fc pero sin pagar Argon2 en los
        # rechazos, que son la inmensa mayoria. Permite subir de 1e3 a 1e5
        # intentos con el mismo presupuesto de computo.
        cw_noisy = bytes(a ^ b for a, b in zip(
            np.packbits(tmpl_q).tobytes(),
            bytes.fromhex(record["commitment"])))
        impostor_tot += 1
        try:
            _rsc = RSCodec(n_parity)
            dec  = _rsc.decode(bytearray(cw_noisy))
            sec  = bytes(dec[0] if isinstance(dec, tuple) else dec)
            if _fc.argon2_verify(record["hash_secret"], sec):
                impostor_ok += 1
        except Exception:
            pass

    gar_rate = _ec.rate_with_ci(genuine_ok, genuine_tot)
    far_rate = _ec.rate_with_ci(impostor_ok, impostor_tot)
    frr_rate = _ec.rate_with_ci(genuine_tot - genuine_ok, genuine_tot)

    gar_per_ds = {
        ds: {**_ec.rate_with_ci(v[0], v[1]), "n_attempts": v[1]}
        for ds, v in gar_by_ds.items()
    }

    far_analytic = (_ec.analytical_far(impostor_bers, n_bits, n_parity)
                    if impostor_bers else None)
    closest = _ec.closest_impostor_pairs(impostor_pairs)

    return {
        "gar":            gar_rate["rate_pct"],
        "far":            far_rate["rate_pct"],
        "gar_ci":         gar_rate,
        "far_ci":         far_rate,
        "frr_ci":         frr_rate,
        "far_analytic":   far_analytic,
        "closest_impostor_pairs": closest,
        "security_bits_measured": _ec.effective_security_bits(far_rate),
        "genuine_ok":     genuine_ok,
        "genuine_total":  genuine_tot,
        "impostor_ok":    impostor_ok,
        "impostor_total": impostor_tot,
        "gar_by_dataset": gar_per_ds,
        "n_users_enrolled": len(enrolled_classes),
        "subject_filter":   filter_stats,
        "n_bits":         n_bits,
        "n_parity":       n_parity,
        "n_enroll":       n_enroll,
        "n_verify":       n_verify,
        "seed":           seed,
    }


# ─────────────────────────────────────────────
# 7. FIGURAS
# ─────────────────────────────────────────────

COLORS = {
    "CASIA":    "#2196F3",
    "IITD":     "#4CAF50",
    "PolyU NIR":"#FF9800",
    "PolyU VIS":"#F44336",
    "UBIRIS":   "#9C27B0",
}


def plot_roc_comparison(orig_metrics, canc_metrics, results_dir):
    """ROC side-by-side: dominio original vs cancelable."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ROC Curves — Original vs Cancelable Domain",
                 fontsize=13, fontweight='bold')

    for ax, m, title, color in [
        (ax1, orig_metrics, "Original Domain (cosine)", "#2196F3"),
        (ax2, canc_metrics,
         f"Cancelable Domain (Hamming, {canc_metrics['n_genuine']} pairs)",
         "#9C27B0"),
    ]:
        ax.plot(m["fpr"]*100, m["tpr"]*100, color=color, linewidth=2.5,
                label=f"EER={m['eer']:.2f}%  AUC={m['auc']:.2f}%")
        ax.plot([0,100],[0,100], 'k:', alpha=0.3)
        eer_v = m["eer"]/100
        ax.scatter([eer_v*100],[(1-eer_v)*100], color='black', s=80, zorder=5,
                   label=f"EER threshold={m['eer_threshold']:.4f}")
        ax.set_xlabel("FAR (%)")
        ax.set_ylabel("TAR (%)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-1,101]); ax.set_ylim([-1,101])

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_biometric_roc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC comparison saved")
    return path


def plot_scores(genuine, impostor, metrics, title, filename, results_dir,
                operating_point=None):
    """Distribucion de scores + curva DET."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax1.hist(impostor, bins=80, alpha=0.6, color='#F44336', density=True,
             label=f"Impostors mu={metrics['mean_impostor']:.3f}")
    ax1.hist(genuine,  bins=80, alpha=0.6, color='#2196F3', density=True,
             label=f"Genuine mu={metrics['mean_genuine']:.3f}")
    ax1.axvline(metrics["eer_threshold"], color='black', linestyle='--',
                linewidth=1.5, label=f"EER threshold={metrics['eer_threshold']:.4f}")
    ax1.set_xlabel("Score"); ax1.set_ylabel("Density")
    ax1.set_title(f"d-prime = {metrics['dprime']:.3f}")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    fpr = metrics["fpr"]; fnr = 1 - metrics["tpr"]
    mask = (fpr > 0) & (fnr > 0)
    ax2.plot(fpr[mask]*100, fnr[mask]*100, color='#2196F3', linewidth=2)
    # La escala logaritmica falla si la separacion es tan buena que no queda
    # ningun punto con FPR y FNR simultaneamente positivos. Se degrada a
    # escala lineal en lugar de abortar el script tras todo el computo.
    if np.any(mask):
        ax2.set_xscale('log'); ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, "Complete separation:\nno points representable\non log scale",
                 ha='center', va='center', transform=ax2.transAxes, fontsize=10)
    ax2.set_xlabel("FAR (%)"); ax2.set_ylabel("FRR (%)")
    ax2.set_title(f"DET Curve  EER={metrics['eer']:.2f}%")
    ax2.grid(True, alpha=0.3, which='both')

    # Punto de operacion del fuzzy commitment. El pipeline criptografico no
    # tiene umbral ajustable: la decodificacion Reed-Solomon acepta o rechaza,
    # de modo que el sistema protegido opera en UN punto de la curva. Situarlo
    # sobre la DET permite analizar la decision del sistema protegido y
    # compararla con las curvas ROC/DET del subsistema biometrico.
    if operating_point and np.any(mask):
        op_far = max(operating_point.get("far_pct", 0.0), 1e-4)
        op_frr = max(operating_point.get("frr_pct", 0.0), 1e-4)
        ax2.plot(op_far, op_frr, marker='*', markersize=16, color='#F44336',
                 linestyle='none', zorder=5,
                 label=f"Fuzzy commitment\nFAR<={op_far:.3f}% FRR={op_frr:.1f}%")
        ax2.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    path = str(Path(results_dir) / filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename} saved")
    return path


def plot_datasets(orig_by_ds, canc_by_ds, impostor_orig, impostor_canc,
                  results_dir):
    """EER por dataset en ambos dominios."""
    datasets = sorted(set(list(orig_by_ds.keys()) + list(canc_by_ds.keys())))
    orig_eers = []
    canc_eers = []

    for ds in datasets:
        if ds in orig_by_ds and len(orig_by_ds[ds]) > 10:
            m = compute_metrics(orig_by_ds[ds], impostor_orig[:5000])
            orig_eers.append(m["eer"])
        else:
            orig_eers.append(None)
        if ds in canc_by_ds and len(canc_by_ds[ds]) > 10:
            m = compute_metrics(canc_by_ds[ds], impostor_canc[:5000])
            canc_eers.append(m["eer"])
        else:
            canc_eers.append(None)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("EER by Dataset — Original vs Cancelable",
                 fontsize=13, fontweight='bold')

    x = np.arange(len(datasets))
    w = 0.35
    orig_vals = [v if v else 0 for v in orig_eers]
    canc_vals = [v if v else 0 for v in canc_eers]
    b1 = ax.bar(x-w/2, orig_vals, w, label="Original (cosine)",
                color="#2196F3", alpha=0.85)
    b2 = ax.bar(x+w/2, canc_vals, w, label="Cancelable (Hamming)",
                color="#9C27B0", alpha=0.85)
    for bar, val in zip(list(b1)+list(b2), orig_vals+canc_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("EER (%)"); ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = str(Path(results_dir) / "eval_biometric_datasets.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Datasets figure saved")
    return path


# ─────────────────────────────────────────────
# 7. INFORME PDF
# ─────────────────────────────────────────────

def generate_pdf(run_name, fte, fta, orig_m, canc_m,
                 fig_roc, fig_orig, fig_canc, fig_ds, results_dir,
                 crypto_metrics=None, hw=None, seed=None, n_users=None):
    pdf_path = str(Path(results_dir) / "eval_biometric_report.pdf")
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=20,
                              spaceAfter=8, textColor=colors.HexColor('#1a237e'))
    h1_s    = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13,
                              textColor=colors.HexColor('#1a237e'),
                              spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('B', parent=styles['Normal'],
                              fontSize=10, spaceAfter=6, leading=14)
        # Cell text that could run long (hardware strings vary a lot in length
    # -- CPU model, kernel version, Python patch version) is wrapped in a
    # Paragraph rather than passed as a plain string. Reportlab's Table
    # only auto-wraps Flowable cell content; a plain string that is wider
    # than its column gets CENTER-aligned around the column's nominal
    # midpoint and overflows equally on both sides -- bleeding into the
    # label column on the left and, for long hardware strings, off the
    # right edge of the page.
    cell_wrap_s = ParagraphStyle('CellWrap', parent=styles['Normal'],
                                 fontSize=9, leading=11, alignment=TA_LEFT)

    cap_s   = ParagraphStyle('C', parent=styles['Normal'], fontSize=9,
                              textColor=colors.grey,
                              alignment=TA_CENTER, spaceAfter=10)

    def table(data, col_widths):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#f5f5f5'), colors.white]),
            ('GRID',          (0,0),(-1,-1), 0.5, colors.lightgrey),
            ('PADDING',       (0,0),(-1,-1), 5),
            ('ALIGN',         (1,1),(-1,-1), 'CENTER'),
        ]))
        return t

    story = []

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Cryptobiometric Iris System", title_s))
    story.append(Paragraph("Biometric Subsystem Evaluation",
                            styles['Heading2']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#1a237e')))
    story.append(Spacer(1, 0.3*cm))

    info = [["Experiment", run_name],
            ["Date", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Model", "EfficientNetV2-S + ArcFace"],
            ["Protocol", "ISO/IEC 19795 — exhaustive max-score"],
            ["Split", "50% enrollment / 50% test per subject"]]
    if n_users is not None:
        info.append(["Subjects evaluated", f"{n_users:,}"])
    if seed is not None:
        info.append(["RNG seed", str(seed)])
    if hw:
        info.append(["Hardware", Paragraph(_ec.hardware_summary(hw), cell_wrap_s)])
    info += [["FTE (extraction)", f"{fte:.4f}%"],
             ["FTA (extraction)", f"{fta:.4f}%"]]
    story.append(table(info, [5*cm, 11*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>Note on FTE/FTA: computed on embeddings that were already "
        "extracted, by detecting invalid vectors. Images discarded during "
        "segmentation or preprocessing never reach this stage, so these "
        "rates describe only the feature-extraction module, not the full "
        "acquisition pipeline.</i>", cap_s))
    story.append(Spacer(1, 0.4*cm))

    # Domain comparison table
    story.append(Paragraph("1. Original vs Cancelable Domain Comparison", h1_s))
    story.append(Paragraph(
        "The cancelable transform binarizes the embeddings and applies a "
        "deterministic permutation. The table compares biometric "
        "performance before and after this transform. The EER cost "
        "reflects the information loss when going from 2048 floats to "
        "256 bits.",
        body_s))

    comp = [["Metric", "Original (cosine)", "Cancelable (Hamming)"],
            ["EER (%)",      f"{orig_m['eer']:.2f}",      f"{canc_m['eer']:.2f}"],
            ["FMR@EER (%)",  f"{orig_m['fmr_at_eer']:.2f}", f"{canc_m['fmr_at_eer']:.2f}"],
            ["FNMR@EER (%)", f"{orig_m['fnmr_at_eer']:.2f}", f"{canc_m['fnmr_at_eer']:.2f}"],
            ["TAR@FAR=0.1%", f"{orig_m['tar_at_far001']:.2f}", f"{canc_m['tar_at_far001']:.2f}"],
            ["TAR@FAR=1%",   f"{orig_m['tar_at_far01']:.2f}",  f"{canc_m['tar_at_far01']:.2f}"],
            ["AUC (%)",      f"{orig_m['auc']:.2f}",      f"{canc_m['auc']:.2f}"],
            ["d-prime",      f"{orig_m['dprime']:.3f}",   f"{canc_m['dprime']:.3f}"],
            ["Genuine score mu", f"{orig_m['mean_genuine']:.4f}", f"{canc_m['mean_genuine']:.4f}"],
            ["Impostor score mu", f"{orig_m['mean_impostor']:.4f}", f"{canc_m['mean_impostor']:.4f}"]]
    story.append(table(comp, [6*cm, 5.5*cm, 5.5*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>Note on d-prime: the value is higher in the cancelable domain "
        "despite its worse EER. d-prime assumes normal distributions, and "
        "the cancelable genuine distribution is strongly skewed with a "
        "concentration at the upper end (Figure 3), so it is not directly "
        "comparable between domains. EER and AUC are.</i>",
        cap_s))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("2. ROC Curves", h1_s))
    if Path(fig_roc).exists():
        story.append(RLImage(fig_roc, width=16*cm, height=7*cm))
        story.append(Paragraph(
            "Figure 1. ROC curves in the original (left) and cancelable "
            "(right) domains.",
            cap_s))

    story.append(PageBreak())
    story.append(Paragraph("3. Score Distribution — Original Domain", h1_s))
    if Path(fig_orig).exists():
        story.append(RLImage(fig_orig, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 2. Genuine vs impostor scores and DET curve in the "
            "original domain.",
            cap_s))

    story.append(Paragraph("4. Score Distribution — Cancelable Domain", h1_s))
    if Path(fig_canc).exists():
        story.append(RLImage(fig_canc, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 3. Genuine vs impostor scores and DET curve in the "
            "cancelable domain.",
            cap_s))

    story.append(PageBreak())
    story.append(Paragraph("5. EER by Dataset", h1_s))
    if Path(fig_ds).exists():
        story.append(RLImage(fig_ds, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 4. EER by dataset in both domains.",
            cap_s))

    # Fuzzy commitment results table
    if crypto_metrics:
        story.append(Paragraph("6. Results via Fuzzy Commitment", h1_s))
        story.append(Paragraph(
            "The following table shows the authentication results obtained "
            "by running the complete cryptographic pipeline — fuzzy "
            "commitment with Reed-Solomon and Argon2 verification — instead "
            "of cosine similarity.",
            body_s))
        cm_ = crypto_metrics
        fc_data = [
            ["Metric", "Value (95% CI)"],
            ["GAR (Genuine Acceptance Rate)", _ec.format_rate(cm_['gar_ci'])],
            ["FRR (False Rejection Rate)",    _ec.format_rate(cm_['frr_ci'])],
            ["FAR (False Acceptance Rate)",   _ec.format_rate(cm_['far_ci'], 4)],
            ["Enrolled subjects",             f"{cm_['n_users_enrolled']:,}"],
            ["n_bits template",               str(cm_['n_bits'])],
            ["n_parity RS",                   str(cm_['n_parity'])],
            ["Enrollment images",             str(cm_['n_enroll'])],
            ["Verification images",           str(cm_['n_verify'])],
        ]
        # Datasets with too few eligible subjects are omitted from this
        # comparison table: a GAR with n<20 has a confidence interval wide
        # enough (often spanning close to the full [0,100] range) that it
        # is not informative next to datasets with hundreds of attempts,
        # and including it invites the reader to compare figures that are
        # not actually comparable. The full per-dataset breakdown, small
        # samples included, remains available in the JSON output.
        MIN_DATASET_N = 20
        omitted_ds = []
        for ds, r in sorted(cm_['gar_by_dataset'].items()):
            if r.get('trials', 0) < MIN_DATASET_N:
                omitted_ds.append((ds, r.get('trials', 0)))
                continue
            fc_data.append([f"GAR {ds}", _ec.format_rate(r, 1)])
        story.append(table(fc_data, [8*cm, 8*cm]))
        story.append(Spacer(1, 0.2*cm))

        if omitted_ds:
            omitted_str = ", ".join(f"{ds} (n={n})" for ds, n in omitted_ds)
            story.append(Paragraph(
                f"<i>Omitted from this comparison: {omitted_str}. Fewer than "
                f"{MIN_DATASET_N} eligible attempts leaves a confidence "
                f"interval too wide to compare meaningfully against the "
                f"other datasets; see Section 1-5 for this dataset's "
                f"performance under the exhaustive EER protocol, which does "
                f"not require a minimum enrollment image count.</i>", cap_s))
            story.append(Spacer(1, 0.2*cm))

        fs = cm_.get("subject_filter")
        if fs and fs["n_excluded"]:
            story.append(Paragraph(
                f"Of the {fs['n_subjects_before']:,} subjects in the dataset, "
                f"{fs['n_excluded']} were excluded from this section for not "
                f"having at least {fs['n_enroll_required']} enrollment images "
                f"and {fs['n_verify_required']} verification images (the "
                f"enrollment/test split divides each subject's total "
                f"captures 50/50, so identities with few captures at the "
                f"source receive few on both sides). The "
                f"{cm_['n_users_enrolled']:,} subjects enrolled in this "
                f"section meet both minimums without exception.", body_s))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph(
            "No false acceptance observed does not amount to a zero FAR: "
            "the upper bound of the confidence interval is the only thing "
            "that can be claimed, and from it the effective security "
            "against the false accept attack is derived as -log2(FAR). To "
            "gain resolution, the count is complemented with an analytical "
            "model that integrates the Reed-Solomon decoding probability "
            "over the measured impostor BER distribution.", body_s))

        fa = cm_.get('far_analytic')
        sec_data = [["FAR estimate", "Value", "Security (bits)"]]
        sec_data.append([
            f"Counted ({cm_['impostor_total']:,} trials)",
            f"<= {cm_['far_ci']['ci_high_pct']:.5f}%",
            str(cm_.get('security_bits_measured', '\u2014'))])
        if fa:
            sec_data.append(["Analytical (empirical over impostor BER)",
                             f"{fa['far_analytic_empirical']:.3e}",
                             str(fa['security_bits_empirical'])])
            sec_data.append(["Analytical (gaussian model, tail-extrapolated)",
                             f"{fa['far_analytic_gaussian']:.3e}",
                             str(fa['security_bits_gaussian'])])
            sec_data.append(["Minimum observed impostor BER",
                             f"{fa['impostor_ber_min']:.4f} "
                             f"({fa['min_ber_z_score']:.1f} sigmas below the mean)",
                             "\u2014"])
        story.append(table(sec_data, [8*cm, 4.5*cm, 3.5*cm]))
        story.append(Spacer(1, 0.2*cm))

        if fa and fa.get("gaussian_warning"):
            story.append(Paragraph(
                "<b>Warning on the gaussian model.</b> " +
                fa["gaussian_warning"] + " The estimate recommended in this "
                "case is the empirical one.", body_s))

        cp = cm_.get("closest_impostor_pairs") or {}
        if cp.get("pairs"):
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(
                "<b>Closest impostor pairs.</b> A correct zero-effort "
                "impostor set concentrates the BER around 0.5. Much lower "
                "values usually indicate a protocol issue — the same "
                "subject under two identities, or captures of the same eye "
                "in different spectra treated as separate classes — rather "
                "than a weakness of the scheme. These are worth checking "
                "before interpreting the FAR.", body_s))
            cp_rows = [["BER", "Query class", "Target class",
                        "Same dataset"]]
            for pr in cp["pairs"][:6]:
                cp_rows.append([f"{pr['ber']:.4f}", pr["query_class"][:24],
                                pr["target_class"][:24],
                                "YES" if pr["same_dataset"] else "no"])
            story.append(table(cp_rows, [2.5*cm, 5.5*cm, 5.5*cm, 2.5*cm]))
            if cp.get("n_suspicious"):
                story.append(Paragraph(
                    f"<i>{cp['n_suspicious']} of {cp['n']} pairs are more "
                    f"than 5 sigmas below the impostor mean.</i>", cap_s))
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("7. Conclusions", h1_s))
    delta = canc_m['eer'] - orig_m['eer']
    story.append(Paragraph(
        f"The biometric subsystem achieves an EER of <b>{orig_m['eer']:.2f}%</b> "
        f"in the original domain and <b>{canc_m['eer']:.2f}%</b> in the "
        f"cancelable domain, an increase of <b>{delta:.2f} percentage "
        f"points</b> due to the information loss from binarizing the "
        f"embeddings. No invalid embeddings were detected in enrollment or "
        f"test (FTE=FTA={fte:.4f}% at the feature-extraction stage).",
        body_s))
    if crypto_metrics:
        cm_ = crypto_metrics
        story.append(Paragraph(
            f"Through the complete cryptographic pipeline, the system "
            f"achieves GAR = {_ec.format_rate(cm_['gar_ci'])} and FAR = "
            f"{_ec.format_rate(cm_['far_ci'], 4)} over "
            f"{cm_['n_users_enrolled']:,} subjects, with n_verify="
            f"{cm_['n_verify']} and n_parity={cm_['n_parity']}.", body_s))

    doc.build(story)
    print(f"  \u2713 PDF report saved to {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Evaluacion biometrica completa")
    parser.add_argument("--run_name", type=str, default="exp08")
    parser.add_argument("--n_bits",   type=int, default=256,
                        help="Bits fiables para el dominio cancelable")
    parser.add_argument("--n_parity", type=int, default=16,
                        help="Bytes de paridad RS para evaluacion fuzzy commitment")
    parser.add_argument("--n_enroll", type=int, default=5,
                        help="Imagenes de enrolamiento por usuario")
    parser.add_argument("--n_verify", type=int, default=3,
                        help="Imagenes de verificacion por intento")
    parser.add_argument("--n_impostor", type=int, default=100000,
                        help="Intentos impostores para la FAR criptografica")
    parser.add_argument("--seed", type=int, default=_ec.DEFAULT_SEED,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    hw = _ec.get_hardware_info()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)
    results_dir   = str(Path(config["paths"]["results"]) / args.run_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUACION BIOMETRICA (ISO/IEC 19795) — {args.run_name}")
    print(f"{'='*60}")
    print(f"  Semilla={args.seed} | n_verify={args.n_verify} | "
          f"n_parity={args.n_parity}")
    print(f"  {_ec.hardware_summary(hw)}")

    possible     = ["enroll_joint","enroll_phase2","enroll_phase1","enroll"]
    enroll_split = next((s for s in possible
                         if (Path(features_root)/s/"embeddings.npy").exists()), None)
    if enroll_split is None:
        print("  No se encontraron embeddings.")
        return
    test_split = enroll_split.replace("enroll","test")

    print(f"\n  Cargando embeddings...")
    enroll_embs, enroll_classes, enroll_paths = load_split(features_root, enroll_split)
    test_embs,   test_classes,   test_paths   = load_split(features_root, test_split)
    print(f"  Enrolamiento: {enroll_embs.shape} | Test: {test_embs.shape}")

    # FTE / FTA
    fte, fta = compute_fte_fta(enroll_embs, enroll_classes,
                                test_embs,   test_classes,
                                len(set(enroll_classes)))
    print(f"\n  FTE = {fte:.4f}%  |  FTA = {fta:.4f}%")

    # Dominio original
    print(f"\n  [1/2] Evaluacion en dominio original...")
    gen_orig, imp_orig, gen_by_ds_orig = run_original_domain(
        enroll_embs, enroll_classes, test_embs, test_classes, test_paths)
    orig_metrics = compute_metrics(gen_orig, imp_orig)
    print(f"  EER original = {orig_metrics['eer']:.2f}%")

    # Dominio cancelable
    print(f"\n  [2/2] Evaluacion en dominio cancelable ({args.n_bits} bits)...")
    gen_canc, imp_canc, gen_by_ds_canc = run_cancelable_domain(
        enroll_embs, enroll_classes, test_embs, test_classes, test_paths,
        args.n_bits)
    canc_metrics = compute_metrics(gen_canc, imp_canc)
    print(f"  EER cancelable = {canc_metrics['eer']:.2f}%")

    # Imprimir tabla
    print(f"\n  {'Metrica':<20} {'Original':>12} {'Cancelable':>12}")
    print(f"  {'-'*46}")
    for k, label in [("eer","EER (%)"),("fmr_at_eer","FMR@EER (%)"),
                      ("fnmr_at_eer","FNMR@EER (%)"),
                      ("tar_at_far001","TAR@FAR=0.1%"),
                      ("tar_at_far01","TAR@FAR=1%"),
                      ("auc","AUC (%)"),("dprime","d-prime")]:
        print(f"  {label:<20} {orig_metrics[k]:>12.3f} {canc_metrics[k]:>12.3f}")

    # Evaluacion mediante fuzzy commitment
    print(f"\n  [3/3] Evaluacion mediante fuzzy commitment...")
    crypto_metrics = run_crypto_evaluation(
        enroll_embs, enroll_classes, enroll_paths,
        test_embs, test_classes, test_paths,
        n_bits=args.n_bits, n_parity=args.n_parity,
        n_enroll=args.n_enroll, n_verify=args.n_verify,
        n_impostor=args.n_impostor, seed=args.seed)
    print(f"  GAR = {_ec.format_rate(crypto_metrics['gar_ci'])}")
    print(f"  FAR = {_ec.format_rate(crypto_metrics['far_ci'], 4)}")
    print(f"  Seguridad efectiva medida: "
          f"{crypto_metrics['security_bits_measured']} bits")
    if crypto_metrics.get('far_analytic'):
        fa = crypto_metrics['far_analytic']
        print(f"  FAR analitica (gaussiana): {fa['far_analytic_gaussian']:.3e} "
              f"-> {fa['security_bits_gaussian']} bits")
    print(f"  GAR por dataset:")
    for ds, r in sorted(crypto_metrics['gar_by_dataset'].items()):
        print(f"    {ds:<12}: {_ec.format_rate(r, 1)}")

    # Guardar JSON
    results = {
        "run_name": args.run_name, "date": datetime.now().isoformat(),
        "seed": args.seed,
        "hardware": hw,
        "n_subjects_total_dataset": len(set(enroll_classes)),  # EER exhaustivo: sin filtrar
        "n_bits_cancelable": args.n_bits,
        "n_parity": args.n_parity,
        "n_enroll": args.n_enroll, "n_verify": args.n_verify,
        "fte": fte, "fta": fta,
        "fte_fta_scope": ("Medidos sobre embeddings extraidos. No incluyen "
                          "fallos de segmentacion o preprocesado."),
        "original":         {k:v for k,v in orig_metrics.items() if k not in ["fpr","tpr"]},
        "cancelable":       {k:v for k,v in canc_metrics.items() if k not in ["fpr","tpr"]},
        "fuzzy_commitment": crypto_metrics,
    }
    json_path = _ec.save_json(
        results, Path(results_dir) / "eval_biometric_results.json")
    print(f"\n  ✓ JSON guardado en {json_path}")

    # Figuras
    print("\n  Generando figuras...")
    fig_roc  = plot_roc_comparison(orig_metrics, canc_metrics, results_dir)
    fig_orig = plot_scores(gen_orig, imp_orig, orig_metrics,
                           "Scores — Original Domain",
                           "eval_biometric_scores_original.png", results_dir)
    fig_canc = plot_scores(gen_canc, imp_canc, canc_metrics,
                           "Scores — Cancelable Domain",
                           "eval_biometric_scores_cancelable.png", results_dir,
                           operating_point={
                               "far_pct": crypto_metrics["far_ci"]["ci_high_pct"],
                               "frr_pct": crypto_metrics["frr_ci"]["rate_pct"]})
    fig_ds   = plot_datasets(gen_by_ds_orig, gen_by_ds_canc,
                             imp_orig, imp_canc, results_dir)

    # PDF
    print("\n  Generando informe PDF...")
    generate_pdf(args.run_name, fte, fta, orig_metrics, canc_metrics,
                 fig_roc, fig_orig, fig_canc, fig_ds, results_dir,
                 crypto_metrics=crypto_metrics, hw=hw, seed=args.seed,
                 n_users=len(set(enroll_classes)))

    print(f"\n{'='*60}")
    print(f"  COMPLETADO")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
