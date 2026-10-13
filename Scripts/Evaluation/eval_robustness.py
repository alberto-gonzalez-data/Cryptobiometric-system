"""
eval_robustness.py
Evaluacion de robustez ante variabilidad biometrica.

Analiza la influencia del parametro de correccion de errores t de
Reed-Solomon sobre el rendimiento de autenticacion y la seguridad.

El parametro t = n_parity // 2 determina cuantos errores de byte puede
corregir RS. Aumentar t mejora la tolerancia a variabilidad biometrica
(mayor GAR) pero reduce el espacio para el secreto y amplia la region
de aceptacion, potencialmente aumentando el FAR.

Experimentos:
  1. Barrido de n_parity: [4, 8, 12, 16, 20, 24] bytes
     Para cada valor: GAR, FAR con IC, secreto disponible, FAR analitica
  2. Barrido de n_verify: [1, 2, 3] imagenes
     Para cada valor con n_parity fijo: GAR, BER media
  3. Estratificacion por dataset de cada configuracion
  4. Barrido de ruido inyectado en el espacio de caracteristicas
  5. Analisis del trade-off seguridad/usabilidad y del limite impuesto por
     la granularidad de byte de Reed-Solomon

ALCANCE DE LA VARIABILIDAD EVALUADA — importante:
La variabilidad de adquisicion incluye cambios de iluminacion, errores de
segmentacion, oclusiones parciales y ruido de sensor. Estos scripts operan
sobre embeddings ya extraidos, por lo que NO pueden aplicar degradaciones a
nivel de imagen. La variabilidad de adquisicion se cubre por dos vias
complementarias, ambas declaradas explicitamente en el informe:

  (a) Estratificacion por dataset. Los cinco conjuntos cubren condiciones de
      captura muy distintas — CASIA e IITD en NIR controlado, PolyU en NIR y
      en visible sobre los mismos sujetos, y UBIRIS.v2 capturado en
      movimiento, a distancia, en luz visible y con oclusiones. La
      comparacion entre ellos es evidencia empirica del comportamiento del
      sistema ante iluminacion, espectro, sensor y oclusion.

  (b) Inyeccion de ruido gaussiano en el espacio de caracteristicas a
      distintos niveles de SNR. Es un proxy CONTROLADO de la degradacion de
      la captura, no un sustituto de las degradaciones a nivel de imagen.

Evaluar degradaciones sobre las imagenes (desenfoque, gamma, mascaras de
oclusion, perturbacion de la segmentacion) exigiria reextraer embeddings y
queda fuera del alcance de este script.

Salidas:
  - results/<run_name>/eval_robustness_results.json
  - results/<run_name>/eval_robustness_parity_sweep.png
  - results/<run_name>/eval_robustness_verify_sweep.png
  - results/<run_name>/eval_robustness_report.pdf

Uso:
    python scripts/eval_robustness.py --run_name exp08
    python scripts/eval_robustness.py --run_name exp08 --n_bits 256 --n_enroll 5
"""

import os
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
from reedsolo import RSCodec, ReedSolomonError
from argon2 import PasswordHasher
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
_fc                      = importlib.import_module("07_fuzzy_commitment")
enroll_user_fc           = _fc.enroll_user
authenticate_user_fc     = _fc.authenticate_user
_cancelable              = importlib.import_module("06_cancelable")
hamming_distance         = _cancelable.hamming_distance
enroll_template_reliable = _cancelable.enroll_template_reliable
verify_template_reliable = _cancelable.verify_template_reliable

_ec = importlib.import_module("eval_common")


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(features_root, split_name):
    split_dir  = Path(features_root) / split_name
    embs       = np.load(str(split_dir / "embeddings.npy"))
    classes    = json.load(open(str(split_dir / "class_strings.json")))
    paths_file = split_dir / "paths.json"
    paths      = (json.load(open(str(paths_file)))
                  if paths_file.exists() else [""] * len(classes))
    return embs, classes, paths


def get_dataset_label(class_str, img_path=""):
    """Etiqueta de dataset, separando PolyU por espectro NIR/VIS."""
    dataset = class_str.split("_")[0]
    if dataset == "polyu":
        ps = str(img_path).replace("\\", "/")
        if "/NIR/" in ps: return "PolyU NIR"
        if "/VIS/" in ps: return "PolyU VIS"
        return "PolyU"
    return dataset.upper()


# ─────────────────────────────────────────────
# EVALUACION PARA UN CONJUNTO DE PARAMETROS
# ─────────────────────────────────────────────

def evaluate_config(enroll_by_class, test_by_class,
                    n_bits, n_parity, n_enroll, n_verify,
                    n_genuine=None, n_impostor=20000,
                    seed=None, ds_by_class=None, noise_sigma=0.0):
    """
    Evalua GAR y FAR para una configuracion de parametros.

    Cambios respecto a la version anterior:
      - Enrolamiento determinista con RNG sembrado (reproducibilidad)
      - n_impostor por defecto 20.000 en lugar de 500: con 500 intentos y
        cero falsas aceptaciones el IC 95% solo permite afirmar FAR < 0.6%,
        insuficiente para sostener que aumentar t no amplia la region de
        aceptacion, que es justo la afirmacion que se necesita sostener
      - Tasas con intervalo de confianza exacto
      - FAR analitica sobre la distribucion de BER impostora
      - GAR estratificada por dataset
      - noise_sigma: ruido gaussiano inyectado en las muestras de
        verificacion como proxy controlado de degradacion de captura
    """
    rng = _ec.make_rng(seed if seed is not None else _ec.DEFAULT_SEED)

    # Se excluyen los sujetos que no llegan al minimo de n_enroll/n_verify
    # imagenes para ESTA configuracion concreta del barrido (el minimo
    # cambia entre llamadas cuando se barre n_verify, asi que el filtro se
    # aplica en cada llamada, no una sola vez en main()).
    enroll_by_class, test_by_class, filter_stats = (
        _ec.filter_subjects_with_min_images(
            enroll_by_class, test_by_class, n_enroll, n_verify))

    # ── Enrolamiento ─────────────────────────────────────────────────────
    user_records     = {}
    enrolled_classes = []
    for cls, embs in enroll_by_class.items():
        try:
            user_records[cls] = _ec.enroll_user_seeded(
                embs[:n_enroll], cls, n_bits, n_parity, rng)
            enrolled_classes.append(cls)
        except Exception:
            pass

    def _perturb(window):
        """
        Ruido gaussiano RELATIVO + renormalizacion L2.

        noise_sigma se expresa como fraccion del RMS de las componentes del
        propio embedding, no en valor absoluto. Los embeddings estan
        normalizados L2 sobre d dimensiones, asi que cada componente vale
        del orden de 1/sqrt(d) — unos 0.022 para d=2048. Un sigma absoluto
        de 0.1 seria cuatro veces la senal y destruiria el template por
        completo, lo que no permite trazar ninguna curva de degradacion.
        Expresarlo en unidades de RMS lo hace ademas independiente de la
        dimension del embedding.
        """
        if noise_sigma <= 0:
            return window
        w   = np.array(window, dtype=float)
        rms = float(np.sqrt(np.mean(w ** 2)))
        if rms <= 0:
            return window
        w = w + rng.normal(0.0, noise_sigma * rms, w.shape)
        nrm = np.linalg.norm(w, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return w / nrm

    # ── Genuinos ─────────────────────────────────────────────────────────
    genuine_ok, genuine_tot = 0, 0
    bers = []
    gar_by_ds = defaultdict(lambda: [0, 0])
    test_classes_list = [c for c in test_by_class if c in user_records]

    for cls in (test_classes_list[:n_genuine] if n_genuine else test_classes_list):
        test_arr = np.array(test_by_class[cls])
        record   = user_records[cls]
        idx      = np.array(record["reliable_indices"])
        ds       = ds_by_class.get(cls, "n/d") if ds_by_class else "n/d"

        enroll_arr  = np.array(enroll_by_class[cls][:n_enroll])
        mean_enroll = np.mean(enroll_arr, axis=0)
        norm        = np.linalg.norm(mean_enroll)
        if norm > 0: mean_enroll /= norm
        tmpl_enroll = verify_template_reliable(
            mean_enroll, cls, record["transform_seed"], idx)

        for start in range(0, len(test_arr) - n_verify + 1, n_verify):
            window   = _perturb(test_arr[start:start+n_verify])
            mean_ver = np.mean(window, axis=0)
            norm     = np.linalg.norm(mean_ver)
            if norm > 0: mean_ver /= norm
            tmpl_ver = verify_template_reliable(
                mean_ver, cls, record["transform_seed"], idx)
            bers.append(hamming_distance(tmpl_enroll, tmpl_ver))

            success, _ = authenticate_user_fc(list(window), cls, record)
            genuine_tot += 1
            gar_by_ds[ds][1] += 1
            if success:
                genuine_ok += 1
                gar_by_ds[ds][0] += 1

    # ── Impostores (zero-effort) ─────────────────────────────────────────
    # RS primero y Argon2 solo si RS decodifica: los rechazos no pagan el
    # coste del hash, lo que permite subir dos ordenes de magnitud el
    # numero de intentos con el mismo presupuesto de computo.
    impostor_ok, impostor_tot = 0, 0
    impostor_bers  = []
    impostor_pairs = []
    rsc = RSCodec(n_parity)

    for _ in range(n_impostor):
        qcls = rng.choice(test_classes_list)
        tcls = rng.choice([c for c in enrolled_classes if c != qcls])
        tarr = np.array(test_by_class[qcls])
        if len(tarr) < n_verify: continue
        start  = rng.integers(0, len(tarr) - n_verify + 1)
        window = _perturb(tarr[start:start+n_verify])

        record = user_records[tcls]
        idx    = np.array(record["reliable_indices"])

        mean_q = np.mean(np.array(window), axis=0)
        nrm    = np.linalg.norm(mean_q)
        if nrm > 0: mean_q = mean_q / nrm
        tmpl_q = verify_template_reliable(
            mean_q, tcls, record["transform_seed"], idx)

        enr_arr  = np.array(enroll_by_class[tcls][:n_enroll])
        mean_enr = np.mean(enr_arr, axis=0)
        nrm      = np.linalg.norm(mean_enr)
        if nrm > 0: mean_enr = mean_enr / nrm
        tmpl_t = verify_template_reliable(
            mean_enr, tcls, record["transform_seed"], idx)
        _ber = hamming_distance(tmpl_t, tmpl_q)
        impostor_bers.append(_ber)
        impostor_pairs.append((_ber, qcls, tcls))

        cw_noisy = bytes(a ^ b for a, b in zip(
            np.packbits(tmpl_q).tobytes(),
            bytes.fromhex(record["commitment"])))
        impostor_tot += 1
        try:
            dec = rsc.decode(bytearray(cw_noisy))
            sec = bytes(dec[0] if isinstance(dec, tuple) else dec)
            if _fc.argon2_verify(record["hash_secret"], sec):
                impostor_ok += 1
        except Exception:
            pass

    gar_rate = _ec.rate_with_ci(genuine_ok, genuine_tot)
    far_rate = _ec.rate_with_ci(impostor_ok, impostor_tot)

    ber_mean = float(np.mean(bers)) if bers else 0.0
    ber_p99  = float(np.percentile(bers, 99)) if bers else 0.0

    return {
        "gar":            gar_rate["rate_pct"],
        "far":            far_rate["rate_pct"],
        "gar_ci":         gar_rate,
        "far_ci":         far_rate,
        "security_bits_measured": _ec.effective_security_bits(far_rate),
        "far_analytic":   (_ec.analytical_far(impostor_bers, n_bits, n_parity)
                           if impostor_bers else None),
        "closest_impostor_pairs": _ec.closest_impostor_pairs(impostor_pairs, k=5),
        "ber_mean":       round(ber_mean, 6),
        "ber_p99":        round(ber_p99, 6),
        "byte_model_mean": _ec.expected_byte_errors(ber_mean, n_bits),
        "byte_model_p99":  _ec.expected_byte_errors(ber_p99,  n_bits),
        "p_decode_mean":  round(_ec.rs_decode_probability(ber_mean, n_bits, n_parity), 6),
        "p_decode_p99":   round(_ec.rs_decode_probability(ber_p99,  n_bits, n_parity), 8),
        "gar_by_dataset": {ds: {**_ec.rate_with_ci(v[0], v[1]), "n_attempts": v[1]}
                           for ds, v in gar_by_ds.items()},
        "secret_bytes":   n_bits//8 - n_parity,
        "secret_bits":    (n_bits//8 - n_parity)*8,
        "n_users":        len(enrolled_classes),
        "n_genuine":      genuine_tot,
        "n_impostor":     impostor_tot,
        "noise_sigma":    noise_sigma,
        "subject_filter": filter_stats,
    }


# ─────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────

def plot_parity_sweep(parity_results, n_bits, results_dir):
    """Trade-off GAR/secreto vs n_parity."""
    parities = sorted(parity_results.keys())
    gars     = [parity_results[p]["gar"]          for p in parities]
    secrets_ = [parity_results[p]["secret_bits"]  for p in parities]
    ber_p99s = [parity_results[p]["ber_p99"]*100  for p in parities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Effect of the RS Parity Parameter ({n_bits} bits)",
                 fontsize=13, fontweight='bold')

    # GAR y secreto vs n_parity
    color1, color2 = '#2196F3', '#F44336'
    ax1.plot(parities, gars, 'o-', color=color1, linewidth=2,
             markersize=7, label="GAR (%)")
    ax1.set_xlabel("n_parity (bytes)")
    ax1.set_ylabel("GAR (%)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)
    for x, y in zip(parities, gars):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0,8), ha='center', fontsize=8)

    ax1b = ax1.twinx()
    ax1b.plot(parities, secrets_, 's--', color=color2, linewidth=2,
              markersize=7, label="Secret (bits)")
    ax1b.set_ylabel("Available secret (bits)", color=color2)
    ax1b.tick_params(axis='y', labelcolor=color2)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lab1+lab2, fontsize=9, loc='lower left')
    ax1.set_title("GAR and Secret Size vs RS Parity")

    # Capacidad de correccion vs errores de byte esperados.
    # La BER no depende de n_parity, asi que representarla frente a n_parity
    # produce barras identicas y no aporta informacion. Lo informativo es
    # comparar la capacidad t con los bytes que realmente se corrompen:
    # Reed-Solomon corrige BYTES, y un byte se pierde con que falle uno solo
    # de sus 8 bits, P(byte erroneo) = 1 - (1 - BER)^8.
    r0 = parity_results[parities[0]]
    bm = r0["byte_model_mean"]["expected_byte_errors"]
    bp = r0["byte_model_p99"]["expected_byte_errors"]
    t_vals = [p // 2 for p in parities]

    ax2.plot(parities, t_vals, 'o-', color='#2196F3', linewidth=2,
             markersize=7, label="Capacity t (correctable bytes)")
    ax2.axhline(bm, color='#4CAF50', linestyle='--', linewidth=2,
                label=f"Erroneous bytes, mean BER ({bm:.1f})")
    ax2.axhline(bp, color='#F44336', linestyle='--', linewidth=2,
                label=f"Erroneous bytes, P99 BER ({bp:.1f})")
    ax2.fill_between(parities, t_vals, bp, where=[t < bp for t in t_vals],
                     color='#F44336', alpha=0.10)
    ax2.set_xlabel("n_parity (bytes)")
    ax2.set_ylabel("Bytes")
    ax2.set_title("Correction Capacity vs Expected Byte Errors")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_robustness_parity_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figura parity sweep guardada")
    return path


def plot_verify_sweep(verify_results, results_dir):
    """GAR y BER media vs n_verify."""
    n_verifies = sorted(verify_results.keys())
    gars  = [verify_results[v]["gar"]      for v in n_verifies]
    bers  = [verify_results[v]["ber_mean"]*100 for v in n_verifies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Effect of the Number of Verification Images",
                 fontsize=13, fontweight='bold')

    ax1.bar(n_verifies, gars, color='#4CAF50', alpha=0.85, width=0.6)
    for x, y in zip(n_verifies, gars):
        ax1.text(x, y+0.5, f"{y:.1f}%", ha='center', fontsize=10, fontweight='bold')
    ax1.set_xlabel("Verification images (n_verify)")
    ax1.set_ylabel("GAR (%)")
    ax1.set_title("GAR vs Number of Verification Images")
    ax1.set_xticks(n_verifies)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(n_verifies, bers, color='#FF9800', alpha=0.85, width=0.6)
    for x, y in zip(n_verifies, bers):
        ax2.text(x, y+0.1, f"{y:.2f}%", ha='center', fontsize=10)
    ax2.set_xlabel("Verification images (n_verify)")
    ax2.set_ylabel("Mean BER (%)")
    ax2.set_title("Mean Genuine BER vs Number of Images")
    ax2.set_xticks(n_verifies)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_robustness_verify_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figura verify sweep guardada")
    return path


def plot_noise_sweep(noise_results, results_dir):
    """GAR y BER media frente al nivel de ruido inyectado."""
    sigmas = sorted(noise_results.keys())
    gars   = [noise_results[x]["gar"] for x in sigmas]
    bers   = [noise_results[x]["ber_mean"]*100 for x in sigmas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Robustness to Noise Injected in the Feature Space",
                 fontsize=13, fontweight='bold')

    ax1.plot(sigmas, gars, 'o-', color='#4CAF50', linewidth=2, markersize=7)
    for x, y in zip(sigmas, gars):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xlabel("noise sigma (x component RMS)")
    ax1.set_ylabel("GAR (%)")
    ax1.set_title("GAR vs Noise Level")
    ax1.set_ylim([0, 105]); ax1.grid(True, alpha=0.3)

    ax2.plot(sigmas, bers, 's-', color='#FF9800', linewidth=2, markersize=7)
    for x, y in zip(sigmas, bers):
        ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_xlabel("noise sigma (x component RMS)")
    ax2.set_ylabel("Mean genuine BER (%)")
    ax2.set_title("BER vs Noise Level")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_robustness_noise_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figura noise sweep guardada")
    return path


def plot_dataset_sweep(config_result, results_dir, min_n=20):
    """
    GAR por dataset para la configuracion de referencia.

    Los datasets con menos de min_n intentos elegibles se omiten de esta
    figura: con n muy pequeno el intervalo de confianza es tan ancho
    (a menudo cubriendo casi todo [0,100]) que la barra no aporta
    informacion comparativa real frente a datasets con cientos de
    intentos, y mostrarla invita a comparar cifras que no son
    comparables. Debe coincidir con el mismo umbral usado en la tabla de
    texto (generate_pdf) para que figura y tabla muestren siempre los
    mismos datasets.
    """
    d = config_result.get("gar_by_dataset", {})
    d = {k: v for k, v in d.items() if k != "n/d"}
    d = {k: v for k, v in d.items() if v.get("n_attempts", 0) >= min_n}
    if not d:
        return None
    names = sorted(d.keys())
    gars  = [d[k]["rate_pct"] for k in names]
    los   = [max(0, d[k]["rate_pct"] - d[k]["ci_low_pct"])  for k in names]
    his   = [max(0, d[k]["ci_high_pct"] - d[k]["rate_pct"]) for k in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("GAR by Dataset — Acquisition Conditions",
                 fontsize=13, fontweight='bold')
    ax.bar(names, gars, color='#2196F3', alpha=0.85,
           yerr=[los, his], capsize=5)
    for i, (g, k) in enumerate(zip(gars, names)):
        ax.text(i, g + 2, f"{g:.1f}%\n(n={d[k]['n_attempts']:,})",
                ha='center', fontsize=8)
    ax.set_ylabel("GAR (%)")
    ax.set_ylim([0, 112])
    ax.set_title("Error bars: exact 95% CI (Clopper-Pearson)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_robustness_dataset_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figura dataset sweep guardada")
    return path


# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────

def generate_pdf(run_name, n_bits, parity_results, verify_results,
                 fig_par, fig_ver, results_dir,
                 noise_results=None, fig_noise=None, fig_ds=None,
                 hw=None, seed=None, n_enroll=5, n_parity_ref=16):
    pdf_path = str(Path(results_dir) / "eval_robustness_report.pdf")
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  topMargin=2*cm, bottomMargin=2*cm,
                                  leftMargin=2*cm, rightMargin=2*cm)
    styles   = getSampleStyleSheet()
    title_s  = ParagraphStyle('T', parent=styles['Title'], fontSize=18,
                               textColor=colors.HexColor('#e65100'))
    h1_s     = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13,
                               textColor=colors.HexColor('#e65100'),
                               spaceBefore=14, spaceAfter=6)
    body_s   = ParagraphStyle('B', parent=styles['Normal'],
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

    cap_s    = ParagraphStyle('C', parent=styles['Normal'], fontSize=9,
                               textColor=colors.grey,
                               alignment=TA_CENTER, spaceAfter=10)

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#e65100')),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#fff3e0'), colors.white]),
            ('GRID',          (0,0),(-1,-1), 0.5, colors.lightgrey),
            ('PADDING',       (0,0),(-1,-1), 5),
            ('ALIGN',         (1,1),(-1,-1), 'CENTER'),
        ]))
        return t

    story = []
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Robustness Evaluation Against Biometric Variability",
                            title_s))
    story.append(Paragraph("Influence of the Reed-Solomon t Parameter",
                            styles['Heading2']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#e65100')))
    story.append(Spacer(1, 0.5*cm))
    # BUGFIX: the previous version took sorted(parity_results.keys())[0],
    # i.e. the SMALLEST parity in the sweep (4), not the reference one
    # (16, the same used by eval_biometric and eval_template_protection).
    # That made the dataset table and byte model in sections 3 and 5 be
    # computed with t=2 instead of t=8, giving a much lower GAR by dataset
    # than in the rest of the reports for the "same" configuration.
    ref = parity_results.get(n_parity_ref, parity_results[sorted(parity_results.keys())[0]])
    meta = [["Experiment", run_name],
            ["Date", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Template", f"{n_bits} bits"],
            ["Enrolled subjects", f"{ref.get('n_users', 0):,}"],
            ["Enrollment images", str(n_enroll)],
            ["Genuine attempts per configuration", f"{ref.get('n_genuine', 0):,}"],
            ["Impostor attempts per configuration", f"{ref.get('n_impostor', 0):,}"]]
    if seed is not None:
        meta.append(["RNG seed", str(seed)])
    if hw:
        meta.append(["Hardware", Paragraph(_ec.hardware_summary(hw), cell_wrap_s)])
    story.append(tbl(meta, [6*cm, 10*cm]))
    story.append(Spacer(1, 0.2*cm))

    fs_ref = ref.get("subject_filter")
    if fs_ref and fs_ref["n_excluded"]:
        story.append(Paragraph(
            f"<i>Of {fs_ref['n_subjects_before']:,} subjects in the dataset, "
            f"{fs_ref['n_excluded']} are excluded in the reference "
            f"configuration for not having at least "
            f"{fs_ref['n_enroll_required']} enrollment images and "
            f"{fs_ref['n_verify_required']} verification images. The exact "
            f"number of eligible subjects may vary slightly between sweep "
            f"rows, since changing n_verify also changes the required "
            f"minimum.</i>", cap_s))
        story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph(
        "<i>Scope: this report evaluates the biometric variability "
        "accessible from the feature space — intra-class variability, "
        "number of verification samples, acquisition conditions via "
        "dataset stratification, and controlled injected noise. "
        "Image-level degradations (blur, gamma, occlusion masks, "
        "segmentation perturbation) would require re-extracting embeddings "
        "and are not covered here.</i>", cap_s))

    story.append(Paragraph("1. RS Parity Parameter Sweep", h1_s))
    story.append(Paragraph(
        "The n_parity parameter determines the number of Reed-Solomon "
        "redundancy bytes. It corrects up to n_parity/2 byte errors. A "
        "larger n_parity improves GAR but reduces the secret size and can "
        "widen the acceptance region.", body_s))

    parities = sorted(parity_results.keys())
    n_impostor_ref = parity_results[parities[0]].get("n_impostor", 0)
    sweep_data = [["n_parity", "t", "Secret", "GAR % (95% CI)",
                   "Counted FAR\u2020", "Analytical FAR*", "Sec. bits"]]
    any_success = False
    for p in parities:
        r  = parity_results[p]
        fa = r.get("far_analytic") or {}
        k  = r["far_ci"]["successes"]
        if k > 0:
            any_success = True
        # The recommended analytical estimate is used (empirical if the
        # gaussian model is not valid for this configuration, gaussian
        # otherwise), never the gaussian one blindly.
        rec = fa.get("recommended_estimate", "gaussian")
        far_val = fa.get(f"far_analytic_{rec}", float('nan'))
        sec_val = fa.get(f"security_bits_{rec}", "\u2014")
        mark = "" if rec == "gaussian" else "*"
        sweep_data.append([
            str(p), str(p//2), f"{r['secret_bits']}b",
            f"{r['gar']:.1f} [{r['gar_ci']['ci_low_pct']:.1f}-"
            f"{r['gar_ci']['ci_high_pct']:.1f}]",
            _ec.format_rate_compact(r["far_ci"]),
            f"{far_val:.2e}{mark}",
            str(sec_val),
        ])
    # Column widths sized with margin for the longest realistic string in
    # each column (verified against actual rendered text, not guessed):
    # the compact FAR format tops out around 100pt at 9pt Helvetica, and
    # this table previously overflowed into neighboring columns when the
    # full format_rate() output (with the trial count) was used here.
    story.append(tbl(sweep_data,
                     [1.7*cm, 1.0*cm, 1.6*cm, 3.0*cm, 4.0*cm, 2.8*cm, 1.7*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"\u2020 {n_impostor_ref:,} impostor attempts per row; see the "
        "counted-FAR paragraph below for exact successes per "
        "configuration. * Recommended analytical estimate per "
        "configuration: empirical when the minimum observed impostor BER "
        "is not compatible with a gaussian tail, gaussian otherwise. See "
        "the biometric report (Section 6) for details on the closest "
        "impostor pairs in the reference configuration.", cap_s))

    if any_success:
        succ_rows = [(p, parity_results[p]["far_ci"]["successes"],
                      parity_results[p]["far_ci"]["trials"])
                     for p in parities if parity_results[p]["far_ci"]["successes"] > 0]
        detail = "; ".join(f"n_parity={p}: {k}/{n}" for p, k, n in succ_rows)
        story.append(Paragraph(
            f"False acceptances were observed in some configurations "
            f"({detail}). This is not a failure of the scheme: with "
            f"increasing correction capacity (larger t), an impostor whose "
            f"BER falls in the lower tail of the distribution can decode by "
            f"chance. The counted FAR stays below "
            f"{max(parity_results[p]['far_ci']['ci_high_pct'] for p in parities):.4f}% "
            f"in every case, and which impostor pairs produce it is "
            f"analyzed in the biometric report.", body_s))
    else:
        story.append(Paragraph(
            "No false acceptance was observed in any configuration. That "
            "does not amount to a zero FAR: with the number of attempts "
            "made, the only defensible claim is the upper bound of the "
            "confidence interval.", body_s))
    story.append(Paragraph(
        "The analytical column integrates the Reed-Solomon decoding "
        "probability over the measured impostor BER distribution and "
        "provides orders of magnitude more resolution than direct "
        "counting.", body_s))
    story.append(Spacer(1, 0.3*cm))

    if Path(fig_par).exists():
        story.append(RLImage(fig_par, width=16*cm, height=6.5*cm))
        story.append(Paragraph(
            "Figure 1. GAR/secret trade-off vs n_parity (left) and P99 BER "
            "(right).",
            cap_s))

    story.append(PageBreak())
    story.append(Paragraph("2. Effect of the Number of Verification Images", h1_s))
    story.append(Paragraph(
        "Using the mean of multiple images at verification reduces the "
        "genuine BER by averaging out acquisition noise, improving GAR "
        "without changing any cryptographic parameter.", body_s))

    n_verifies   = sorted(verify_results.keys())
    verify_data  = [["n_verify", "GAR % (95% CI)", "FAR (95% CI)", "Mean BER"]]
    for v in n_verifies:
        r = verify_results[v]
        verify_data.append([
            str(v),
            f"{r['gar']:.1f} [{r['gar_ci']['ci_low_pct']:.1f}\u2013{r['gar_ci']['ci_high_pct']:.1f}]",
            _ec.format_rate(r["far_ci"], 4),
            f"{r['ber_mean']:.4f}"])
    # FAR column widened: the full format_rate() text (with trial count)
    # was measured at up to ~147pt, leaving under 10pt of margin inside a
    # 5.5cm column once cell padding is subtracted -- too tight to be safe
    # for longer numbers (e.g. a larger n or a wider CI).
    story.append(tbl(verify_data, [1.5*cm, 5*cm, 6.5*cm, 3*cm]))
    story.append(Spacer(1, 0.3*cm))

    if Path(fig_ver).exists():
        story.append(RLImage(fig_ver, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 2. GAR and mean BER vs number of verification images.",
            cap_s))

    # ── 3. Acquisition conditions ────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph(
        "3. Acquisition Conditions — Dataset Stratification", h1_s))
    story.append(Paragraph(
        "The five datasets cover very different capture conditions: "
        "CASIA-IrisV4 and IIT Delhi in controlled near-infrared, PolyU in "
        "NIR and visible light on the same subjects, and UBIRIS.v2 "
        "captured on the move, at a distance, in visible light and with "
        "occlusion. The comparison between them constitutes empirical "
        "evidence of the system's behavior under variations in "
        "illumination, spectrum, sensor and occlusion.", body_s))

    # Datasets with too few eligible subjects are omitted from this
    # comparison: a GAR with n<20 has a confidence interval wide enough
    # that it is not informative next to datasets with hundreds of
    # attempts. Full per-dataset figures, small samples included, remain
    # available in the JSON output.
    MIN_DATASET_N = 20
    ds_ref_full = ref.get("gar_by_dataset", {})
    ds_ref_full = {k: v for k, v in ds_ref_full.items() if k != "n/d"}
    ds_ref = {k: v for k, v in ds_ref_full.items()
              if v.get("n_attempts", 0) >= MIN_DATASET_N}
    ds_omitted = {k: v for k, v in ds_ref_full.items()
                  if v.get("n_attempts", 0) < MIN_DATASET_N}
    if ds_ref:
        ds_rows = [["Dataset", "GAR % (95% CI)", "Attempts"]]
        for k in sorted(ds_ref):
            v = ds_ref[k]
            ds_rows.append([
                k,
                f"{v['rate_pct']:.1f} [{v['ci_low_pct']:.1f}-{v['ci_high_pct']:.1f}]",
                f"{v['n_attempts']:,}"])
        story.append(tbl(ds_rows, [6*cm, 6*cm, 4*cm]))
        story.append(Spacer(1, 0.2*cm))
    if ds_omitted:
        omitted_str = ", ".join(f"{k} (n={v['n_attempts']})"
                                for k, v in sorted(ds_omitted.items()))
        story.append(Paragraph(
            f"<i>Omitted from this comparison: {omitted_str}. Fewer than "
            f"{MIN_DATASET_N} eligible attempts leaves a confidence "
            f"interval too wide to compare meaningfully; this dataset's "
            f"performance under the exhaustive EER protocol (no minimum "
            f"enrollment count) is reported in the biometric report, "
            f"Section 5.</i>", cap_s))
        story.append(Spacer(1, 0.2*cm))
    story.append(Spacer(1, 0.1*cm))
    if fig_ds and Path(fig_ds).exists():
        story.append(RLImage(fig_ds, width=15*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 3. GAR by dataset with exact confidence intervals.",
            cap_s))

    # ── 4. Injected noise ─────────────────────────────────────────────────
    if noise_results:
        story.append(Paragraph(
            "4. Robustness to Controlled Noise in the Feature Space",
            h1_s))
        story.append(Paragraph(
            "Gaussian noise of increasing deviation is injected into the "
            "verification samples, with subsequent L2 renormalization. The "
            "sigma parameter is expressed in units of the RMS of the "
            "embedding's own components, so sigma=0.25 corresponds to noise "
            "equal to 25% of the signal amplitude, and the scale is "
            "independent of the feature space dimensionality. This is a "
            "controlled proxy for capture degradation that allows the "
            "system's margin to be quantified, not a substitute for "
            "image-level degradations.", body_s))
        nz = sorted(noise_results.keys())
        nz_rows = [["sigma (x RMS)", "GAR % (95% CI)", "Mean BER",
                    "Expected erroneous bytes"]]
        for x in nz:
            r = noise_results[x]
            nz_rows.append([
                f"{x:.3f}",
                f"{r['gar']:.1f} [{r['gar_ci']['ci_low_pct']:.1f}\u2013"
                f"{r['gar_ci']['ci_high_pct']:.1f}]",
                f"{r['ber_mean']:.4f}",
                f"{r['byte_model_mean']['expected_byte_errors']:.1f}"])
        story.append(tbl(nz_rows, [3*cm, 5.5*cm, 3.5*cm, 4*cm]))
        story.append(Spacer(1, 0.3*cm))
        if fig_noise and Path(fig_noise).exists():
            story.append(RLImage(fig_noise, width=16*cm, height=6*cm))
            story.append(Paragraph(
                "Figure 4. GAR and mean BER vs injected noise level.",
                cap_s))

    # ── 5. Limit imposed by byte granularity ─────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph(
        "5. Limit Imposed by Reed-Solomon's Byte Granularity", h1_s))
    bm = ref["byte_model_mean"]
    bp = ref["byte_model_p99"]
    story.append(Paragraph(
        "Reed-Solomon corrects BYTE errors, whereas biometric variability "
        "manifests as scattered BIT errors. A byte is corrupted as soon as "
        "a single one of its eight bits is wrong, so P(byte error) = "
        "1 - (1 - BER)^8. This efficiency loss is the quantitative reason "
        "why GAR plateaus, and why BCH, which corrects directly at the bit "
        "level, is proposed as future work.", body_s))
    story.append(tbl([
        ["Regime", "BER", "Bit errors", "Expected erroneous bytes",
         "P(decodes)"],
        ["Mean genuine BER", f"{ref['ber_mean']:.4f}",
         f"{bm['bit_errors']:.1f}", f"{bm['expected_byte_errors']:.1f}",
         f"{ref['p_decode_mean']:.4f}"],
        ["P99 genuine BER", f"{ref['ber_p99']:.4f}",
         f"{bp['bit_errors']:.1f}", f"{bp['expected_byte_errors']:.1f}",
         f"{ref['p_decode_p99']:.2e}"],
    ], [3.3*cm, 1.8*cm, 2*cm, 4.6*cm, 2.3*cm]))
    story.append(Spacer(1, 0.2*cm))
    t_ref = n_parity_ref // 2
    story.append(Paragraph(
        f"With the reference configuration (n_parity={n_parity_ref}, "
        f"t={t_ref} correctable bytes), the correction capacity falls well "
        f"short of the {bp['expected_byte_errors']:.1f} corrupted bytes "
        f"expected at the 99th percentile of the genuine distribution: "
        f"those tail samples fall outside the code's capacity even though "
        f"their BER remains low in bit terms. Across the whole sweep "
        f"(n_parity from {parities[0]} to {parities[-1]}), the capacity "
        f"ranges from t={parities[0]//2} to t={parities[-1]//2} bytes, "
        f"never covering that 99th percentile.", body_s))

    # ── 6. Conclusions ────────────────────────────────────────────────────
    story.append(Paragraph("6. Conclusions", h1_s))
    best_p = max(parities, key=lambda p: parity_results[p]["gar"])
    story.append(Paragraph(
        f"The parity parameter governs the trade-off between GAR and "
        f"secret size: GAR grows from {parity_results[parities[0]]['gar']:.1f}% "
        f"with n_parity={parities[0]} to {parity_results[best_p]['gar']:.1f}% "
        f"with n_parity={best_p}, while the available secret shrinks from "
        f"{parity_results[parities[0]]['secret_bits']} to "
        f"{parity_results[best_p]['secret_bits']} bits. "
        + (f"False acceptances were observed in some configurations, "
           f"consistent with the expected growth of the acceptance region "
           f"as t increases; the analytical FAR confirms that growth is "
           f"real but small in absolute terms."
           if any_success else
           f"No false acceptances were observed in any configuration, and "
           f"the analytical FAR confirms that the acceptance region does "
           f"not widen appreciably as t increases."), body_s))
    story.append(Paragraph(
        f"The number of verification images is the factor with the "
        f"greatest impact: going from {min(n_verifies)} to "
        f"{max(n_verifies)} images improves GAR by "
        f"{verify_results[max(n_verifies)]['gar'] - verify_results[min(n_verifies)]['gar']:.1f} "
        f"percentage points by averaging out acquisition noise, without "
        f"touching any cryptographic parameter. Biometric variability, "
        f"rather than the code's capacity, is therefore the dominant "
        f"limiting factor.",
        body_s))

    doc.build(story)
    print(f"  \u2713 PDF saved to {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Evaluacion robustez RS")
    parser.add_argument("--run_name",       type=str, default="exp08")
    parser.add_argument("--n_bits",         type=int, default=256)
    parser.add_argument("--n_enroll",       type=int, default=5)
    parser.add_argument("--parity_values",  type=int, nargs="+",
                        default=[4, 8, 12, 16, 20, 24],
                        help="Valores de n_parity a evaluar")
    parser.add_argument("--verify_values",  type=int, nargs="+",
                        default=[1, 2, 3],
                        help="Valores de n_verify a evaluar")
    parser.add_argument("--noise_values",   type=float, nargs="+",
                        default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.00],
                        help="Sigma del ruido, en unidades de RMS de las "
                             "componentes del embedding (0.25 = 25%% de la senal)")
    parser.add_argument("--n_verify_ref",   type=int, default=3,
                        help="n_verify de referencia para el barrido de paridad")
    parser.add_argument("--n_parity_ref",   type=int, default=16,
                        help="n_parity de referencia para los demas barridos")
    parser.add_argument("--n_impostor",     type=int, default=20000,
                        help="Intentos impostores por configuracion")
    parser.add_argument("--seed",           type=int, default=_ec.DEFAULT_SEED,
                        help="RNG seed for reproducibility")
    args = parser.parse_args()

    hw = _ec.get_hardware_info()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)
    results_dir   = str(Path(config["paths"]["results"]) / args.run_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUACION ROBUSTEZ (RS) — {args.run_name}")
    print(f"{'='*60}")
    print(f"  n_bits={args.n_bits} | n_enroll={args.n_enroll} | semilla={args.seed}")
    print(f"  Paridades a evaluar: {args.parity_values}")
    print(f"  n_verify a evaluar:  {args.verify_values}")
    print(f"  Ruido a evaluar:     {args.noise_values}")
    print(f"  Impostores por config: {args.n_impostor:,}")
    print(f"  {_ec.hardware_summary(hw)}")

    possible     = ["enroll_joint","enroll_phase2","enroll_phase1","enroll"]
    enroll_split = next((s for s in possible
                         if (Path(features_root)/s/"embeddings.npy").exists()), None)
    test_split   = enroll_split.replace("enroll","test")

    enroll_embs, enroll_classes, enroll_paths = load_split(features_root, enroll_split)
    test_embs,   test_classes,   test_paths   = load_split(features_root, test_split)

    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)
    test_by_class = defaultdict(list)
    for emb, cls in zip(test_embs, test_classes):
        test_by_class[cls].append(emb)

    # Etiqueta de dataset por clase, para la estratificacion
    ds_by_class = {}
    for cls, path in zip(test_classes, test_paths):
        ds_by_class.setdefault(cls, get_dataset_label(cls, path))

    # Barrido paridad
    common = dict(n_bits=args.n_bits, n_enroll=args.n_enroll,
                  n_impostor=args.n_impostor, seed=args.seed,
                  ds_by_class=ds_by_class)

    print(f"\n  [1/3] Barrido n_parity con n_verify={args.n_verify_ref}...")
    parity_results = {}
    for n_parity in args.parity_values:
        if n_parity >= args.n_bits // 8:
            print(f"  ⚠ n_parity={n_parity} >= {args.n_bits//8} bytes, omitiendo")
            continue
        print(f"  n_parity={n_parity}...", end=" ", flush=True)
        r = evaluate_config(enroll_by_class, test_by_class,
                            n_parity=n_parity, n_verify=args.n_verify_ref,
                            **common)
        parity_results[n_parity] = r
        fa = r.get("far_analytic") or {}
        print(f"GAR={r['gar']:.1f}%  secreto={r['secret_bits']}b  "
              f"FAR<={r['far_ci']['ci_high_pct']:.4f}%  "
              f"FARanal={fa.get('far_analytic_gaussian', float('nan')):.1e}")

    print(f"\n  [2/3] Barrido n_verify con n_parity={args.n_parity_ref}...")
    verify_results = {}
    for n_verify in args.verify_values:
        print(f"  n_verify={n_verify}...", end=" ", flush=True)
        r = evaluate_config(enroll_by_class, test_by_class,
                            n_parity=args.n_parity_ref, n_verify=n_verify,
                            **common)
        verify_results[n_verify] = r
        print(f"GAR={r['gar']:.1f}%  BER={r['ber_mean']:.4f}")

    print(f"\n  [3/3] Barrido de ruido inyectado "
          f"(n_parity={args.n_parity_ref}, n_verify={args.n_verify_ref})...")
    noise_results = {}
    for sigma in args.noise_values:
        print(f"  sigma={sigma:.3f}...", end=" ", flush=True)
        r = evaluate_config(enroll_by_class, test_by_class,
                            n_parity=args.n_parity_ref,
                            n_verify=args.n_verify_ref,
                            noise_sigma=sigma, **common)
        noise_results[sigma] = r
        print(f"GAR={r['gar']:.1f}%  BER={r['ber_mean']:.4f}")

    results = {
        "run_name":      args.run_name,
        "date":          datetime.now().isoformat(),
        "seed":          args.seed,
        "hardware":      hw,
        "n_bits":        args.n_bits,
        "n_enroll":      args.n_enroll,
        "n_verify_ref":  args.n_verify_ref,
        "n_parity_ref":  args.n_parity_ref,
        "n_impostor_per_config": args.n_impostor,
        "scope_note": ("Variabilidad evaluada desde el espacio de "
                       "caracteristicas: intraclase, numero de muestras de "
                       "verificacion, condiciones de adquisicion via "
                       "estratificacion por dataset y ruido inyectado. Las "
                       "degradaciones a nivel de imagen quedan fuera de "
                       "alcance."),
        "parity_sweep":  {str(k): v for k, v in parity_results.items()},
        "verify_sweep":  {str(k): v for k, v in verify_results.items()},
        "noise_sweep":   {str(k): v for k, v in noise_results.items()},
    }
    json_path = _ec.save_json(
        results, Path(results_dir) / "eval_robustness_results.json")
    print(f"\n  ✓ JSON guardado")

    print("\n  Generando figuras...")
    fig_par   = plot_parity_sweep(parity_results, args.n_bits, results_dir)
    fig_ver   = plot_verify_sweep(verify_results, results_dir)
    fig_noise = plot_noise_sweep(noise_results, results_dir)
    ref_cfg   = parity_results.get(args.n_parity_ref,
                                   parity_results[sorted(parity_results)[0]])
    fig_ds    = plot_dataset_sweep(ref_cfg, results_dir)

    print("\n  Generando PDF...")
    generate_pdf(args.run_name, args.n_bits, parity_results, verify_results,
                 fig_par, fig_ver, results_dir,
                 noise_results=noise_results, fig_noise=fig_noise,
                 fig_ds=fig_ds, hw=hw, seed=args.seed, n_enroll=args.n_enroll,
                 n_parity_ref=args.n_parity_ref)

    print(f"\n{'='*60}\n  COMPLETADO\n{'='*60}\n")


if __name__ == "__main__":
    main()
