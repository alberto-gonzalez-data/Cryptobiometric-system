"""
eval_template_protection.py
Evaluacion de las propiedades de proteccion de templates.

Evalua las tres propiedades fundamentales de proteccion biometrica
definidas en ISO/IEC 24745: no-invertibilidad, revocabilidad y unlinkability.

1. No-invertibilidad:
   Se evalua mediante ataques de inversion de template y de reconstruccion
   del secreto bajo un escenario de compromiso total del template. Se
   ejecutan ataques reales, no solo estadisticos descriptivos:
     - Ataque de inversion de template (con y sin el transform_seed)
     - Coste analitico de reconstruccion del secreto y fuga estructural
       del commitment (coset del codigo Reed-Solomon)
     - Entropia efectiva via grados de libertad de Daugman
   Los estadisticos del commitment (fraccion de unos, entropia) se
   conservan como comprobacion de sanidad, pero se marcan explicitamente
   como garantizados por construccion: delta = xc XOR c con c derivada de
   un secreto aleatorio es uniforme independientemente de la biometria,
   de modo que esa metrica no puede fallar y no demuestra no-invertibilidad.

2. Revocabilidad:
   Para cada usuario se generan dos templates con seeds distintos (seed_A, seed_B).
   La distancia de Hamming entre ambos templates debe ser ~0.5, demostrando
   que templates revocados no tienen correlacion con los nuevos.

3. Unlinkability:
   Se compara la distribucion de similitudes entre templates relacionados
   (mismo usuario, distinto seed) y no relacionados (usuarios distintos).
   Se reporta la metrica D_sys del marco de Gomez-Barrero et al. (IEEE
   TIFS 2018), que es el estandar de la literatura y tiene recorrido
   [0, 1]. La version anterior de este script llamaba D_sys a un tamano
   del efecto (Cohen's d), que es otra magnitud con otra escala: se
   conserva con su nombre correcto para poder comparar.

   Ademas se evalua el enlace a traves del REGISTRO COMPLETO almacenado,
   no solo del vector de bits: los indices fiables se derivan del
   embedding y no del seed, por lo que pueden constituir un canal de
   enlace que la comparacion de templates no detecta.

Salidas:
  - results/<run_name>/eval_template_protection_results.json
  - results/<run_name>/eval_template_protection_revocability.png
  - results/<run_name>/eval_template_protection_unlinkability.png
  - results/<run_name>/eval_template_protection_report.pdf

Uso:
    python scripts/eval_template_protection.py --run_name exp08
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
import yaml
from datetime import datetime
from scipy.stats import entropy as scipy_entropy

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

_fc                  = importlib.import_module("07_fuzzy_commitment")
generate_commitment  = _fc.generate_commitment
verify_commitment    = _fc.verify_commitment

_ec = importlib.import_module("eval_common")


# ─────────────────────────────────────────────
# 1. CONFIGURACION
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(features_root, split_name):
    split_dir  = Path(features_root) / split_name
    embeddings = np.load(str(split_dir / "embeddings.npy"))
    classes    = json.load(open(str(split_dir / "class_strings.json")))
    return embeddings, classes


# ─────────────────────────────────────────────
# 2. NO-INVERTIBILIDAD
# ─────────────────────────────────────────────

def analyze_non_invertibility(enroll_embs, enroll_classes,
                               n_bits, n_parity, n_users=100, rng=None):
    """
    Analiza la no-invertibilidad del commitment.

    El commitment delta = xc XOR c mezcla el template biometrico con el
    secreto codificado. Sin conocer uno de los dos no se puede recuperar
    el otro. Se analizan:
      - Distribucion de bits del commitment (debe ser uniforme ~50% unos)
      - Entropia del commitment (debe ser maxima ~1.0 bits/bit)
      - Correlacion entre commitment y template original
    """
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    classes = list(enroll_by_class.keys())[:n_users]
    bit_fractions   = []
    entropies       = []
    correlations    = []

    rng = rng if rng is not None else _ec.make_rng()

    for cls in classes:
        embs = enroll_by_class[cls]
        seed = rng.bytes(32).hex()
        tmpl, indices = enroll_template_reliable(embs, cls, seed, n_bits)

        commit_data = generate_commitment(tmpl, n_parity)
        commit_bytes = bytes.fromhex(commit_data["commitment"])
        commit_bits  = np.unpackbits(np.frombuffer(commit_bytes, dtype=np.uint8))

        # Fraccion de unos (debe ser ~0.5)
        bit_fractions.append(float(np.mean(commit_bits)))

        # Entropia (debe ser ~1.0)
        p1 = np.mean(commit_bits)
        p0 = 1 - p1
        if 0 < p0 < 1:
            ent = -p1*np.log2(p1+1e-12) - p0*np.log2(p0+1e-12)
        else:
            ent = 0.0
        entropies.append(ent)

        # Correlacion entre commitment y template original
        tmpl_float = tmpl.astype(float)
        commit_float = commit_bits[:n_bits].astype(float)
        corr = float(np.corrcoef(tmpl_float, commit_float)[0,1])
        correlations.append(abs(corr))

    return {
        "bit_fraction_mean":  round(float(np.mean(bit_fractions)),  6),
        "bit_fraction_std":   round(float(np.std(bit_fractions)),   6),
        "entropy_mean":       round(float(np.mean(entropies)),       6),
        "entropy_std":        round(float(np.std(entropies)),        6),
        "correlation_mean":   round(float(np.mean(correlations)),    6),
        "correlation_std":    round(float(np.std(correlations)),     6),
        "n_users_analyzed":   len(classes),
        "caveat": ("Estos tres estadisticos estan garantizados por "
                   "construccion: delta = xc XOR c con c derivada de un "
                   "secreto aleatorio es uniforme sea cual sea la biometria. "
                   "Sirven como comprobacion de sanidad, no como evidencia "
                   "de no-invertibilidad. La evidencia esta en los ataques "
                   "de la seccion siguiente y en los grados de libertad."),
    }


# ─────────────────────────────────────────────
# 3. REVOCABILIDAD
# ─────────────────────────────────────────────

def analyze_revocability(enroll_embs, enroll_classes,
                          test_embs, test_classes,
                          n_bits, n_users=200):
    """
    Evalua la revocabilidad generando dos templates del mismo usuario
    con seeds distintos y midiendo su distancia de Hamming.

    Para revocabilidad perfecta: BER(seed_A, seed_B) ~= 0.5
    Esto demuestra que templates revocados no tienen correlacion
    con los nuevos templates generados.

    La metrica 'mismo usuario, mismo seed' compara el template de enrolamiento
    contra templates generados desde imagenes de test individuales usando los
    MISMOS indices fiables — esto mide la variabilidad intraclase real del sistema.
    """
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    test_by_class = defaultdict(list)
    for emb, cls in zip(test_embs, test_classes):
        test_by_class[cls].append(emb)

    classes        = list(enroll_by_class.keys())[:n_users]
    bers_same_seed = []  # BER enrolamiento vs verificacion, mismo seed
    bers_diff_seed = []  # BER mismo usuario, seeds distintos
    bers_diff_user = []  # BER usuarios distintos (referencia)

    rng = np.random.default_rng(42)

    for cls in classes:
        embs   = enroll_by_class[cls]
        seed_A = secrets.token_hex(32)
        seed_B = secrets.token_hex(32)

        # Template de enrolamiento con seed_A (usa todos los embeddings)
        tmpl_A, idx_A = enroll_template_reliable(embs, cls, seed_A, n_bits)
        # Template de enrolamiento con seed_B (mismos embeddings, seed distinto)
        tmpl_B, _     = enroll_template_reliable(embs, cls, seed_B, n_bits)

        # Mismo usuario, mismo seed: comparar enrolamiento vs imagenes de test
        # usando los MISMOS indices — mide variabilidad intraclase real
        if cls in test_by_class:
            for test_emb in test_by_class[cls][:3]:  # max 3 imagenes de test
                ver_tmpl = verify_template_reliable(test_emb, cls, seed_A, idx_A)
                bers_same_seed.append(hamming_distance(tmpl_A, ver_tmpl))

        # Mismo usuario, seeds distintos → debe ser ~0.5
        bers_diff_seed.append(hamming_distance(tmpl_A, tmpl_B))

    # Usuarios distintos como referencia
    class_list = list(enroll_by_class.keys())
    for _ in range(min(500, len(classes)*5)):
        c1   = rng.choice(class_list)
        c2   = rng.choice([c for c in class_list if c != c1])
        seed = secrets.token_hex(32)
        t1, _ = enroll_template_reliable(enroll_by_class[c1], c1, seed, n_bits)
        t2, _ = enroll_template_reliable(enroll_by_class[c2], c2, seed, n_bits)
        bers_diff_user.append(hamming_distance(t1, t2))

    return {
        "same_user_same_seed": {
            "ber_mean": round(float(np.mean(bers_same_seed)), 6),
            "ber_std":  round(float(np.std(bers_same_seed)),  6),
            "n":        len(bers_same_seed),
            "note":     "Variabilidad intraclase real (enrol vs test, mismos indices)"
        },
        "same_user_diff_seed": {
            "ber_mean": round(float(np.mean(bers_diff_seed)), 6),
            "ber_std":  round(float(np.std(bers_diff_seed)),  6),
            "n":        len(bers_diff_seed),
            "note":     "Debe ser ~0.5 para garantizar revocabilidad"
        },
        "diff_user": {
            "ber_mean": round(float(np.mean(bers_diff_user)), 6),
            "ber_std":  round(float(np.std(bers_diff_user)),  6),
            "n":        len(bers_diff_user),
        },
    }, bers_same_seed, bers_diff_seed, bers_diff_user


# ─────────────────────────────────────────────
# 4. UNLINKABILITY
# ─────────────────────────────────────────────

def analyze_unlinkability(enroll_embs, enroll_classes, n_bits, n_users=200,
                           rng=None):
    """
    Evalua unlinkability midiendo si templates del mismo usuario con
    distintos seeds son distinguibles de templates de usuarios distintos.

    Metrica: D_sys = |mu_related - mu_unrelated| / sigma_pooled
    Para unlinkability perfecta: D_sys ~ 0 (distribuciones indistinguibles)

    Los templates con el mismo seed son distinguibles (BER < 0.5),
    pero con seeds distintos deben ser indistinguibles (BER ~ 0.5).
    """
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    classes    = list(enroll_by_class.keys())[:n_users]
    rng        = rng if rng is not None else _ec.make_rng()
    class_list = list(enroll_by_class.keys())

    # Distribucion relacionada: mismo usuario, seeds distintos
    bers_related = []
    for cls in classes:
        seed_A = rng.bytes(32).hex()
        seed_B = rng.bytes(32).hex()
        tA, _ = enroll_template_reliable(enroll_by_class[cls], cls, seed_A, n_bits)
        tB, _ = enroll_template_reliable(enroll_by_class[cls], cls, seed_B, n_bits)
        bers_related.append(hamming_distance(tA, tB))

    # Distribucion no relacionada: usuarios distintos, seeds distintos
    bers_unrelated = []
    for _ in range(len(bers_related)):
        c1    = rng.choice(class_list)
        c2    = rng.choice([c for c in class_list if c != c1])
        seed1 = rng.bytes(32).hex()
        seed2 = rng.bytes(32).hex()
        t1, _ = enroll_template_reliable(enroll_by_class[c1], c1, seed1, n_bits)
        t2, _ = enroll_template_reliable(enroll_by_class[c2], c2, seed2, n_bits)
        bers_unrelated.append(hamming_distance(t1, t2))

    mu_r  = np.mean(bers_related)
    mu_u  = np.mean(bers_unrelated)
    sig_r = np.std(bers_related)
    sig_u = np.std(bers_unrelated)

    # Metrica estandar (Gomez-Barrero et al., IEEE TIFS 2018)
    gb, gb_centers, gb_local = _ec.unlinkability_gomez_barrero(
        bers_related, bers_unrelated)

    return {
        "related_templates": {
            "ber_mean": round(float(mu_r),  6),
            "ber_std":  round(float(sig_r), 6),
            "n":        len(bers_related),
            "note":     "Same user, different seeds"
        },
        "unrelated_templates": {
            "ber_mean": round(float(mu_u),  6),
            "ber_std":  round(float(sig_u), 6),
            "n":        len(bers_unrelated),
            "note":     "Different users, different seeds"
        },
        "gomez_barrero":  gb,
        "D_sys":          gb["D_sys_gomez_barrero"],
        "verdict":        gb["verdict"],
        "effect_size_cohens_d": _ec.cohens_d(bers_related, bers_unrelated),
        "note": ("D_sys segun Gomez-Barrero et al. (IEEE TIFS 2018), "
                 "recorrido [0,1]; 0 = unlinkable. El valor 'effect_size_"
                 "cohens_d' es la magnitud que versiones anteriores de este "
                 "script reportaban erroneamente como D_sys."),
    }, bers_related, bers_unrelated



# ─────────────────────────────────────────────
# 4b. ATAQUES BAJO COMPROMISO TOTAL DE LA BD
# ─────────────────────────────────────────────

def run_security_attacks(enroll_by_class, n_bits, n_parity, n_users, rng,
                          emb_dim=None, argon2_ms=14.0):
    """
    Ejecuta los ataques de inversion, reconstruccion del secreto y
    cross-matching bajo un modelo de compromiso total: el adversario
    dispone de la base
    de datos completa y de conocimiento total del algoritmo, pero no de
    una muestra biometrica valida.

    Ataques ejecutados:
      1. Inversion de template
      2. Reconstruccion del secreto (analitico)
      3. Cross-matching tras revocacion
      4. Enlace por indices fiables

    Los resultados se interpretan en el informe. Conviene subrayar que un
    resultado desfavorable no invalida el esquema criptografico: identifica
    que la proteccion depende de donde se custodie el transform_seed.
    """
    out = {}

    print("    [a] Inversion de template...")
    out["template_inversion"] = _ec.attack_template_inversion(
        enroll_by_class, n_bits, min(n_users, 50), rng)

    print("    [b] Coste de reconstruccion del secreto...")
    out["secret_reconstruction"] = _ec.secret_reconstruction_cost(
        n_bits, n_parity, argon2_ms)

    print("    [c] Cross-matching entre templates revocados...")
    out["cross_matching"] = _ec.attack_cross_matching(
        enroll_by_class, n_bits, n_parity, min(n_users, 50), rng)

    print("    [d] Enlace por indices fiables...")
    out["index_linkage"] = _ec.attack_index_linkage(
        enroll_by_class, n_bits, min(n_users, 100), rng, emb_dim=emb_dim)

    # Veredicto agregado
    inv_ok  = out["template_inversion"]["ber_with_stored_seed"] < 0.05
    xm_ok   = out["cross_matching"]["informed_cross_match"]["rate_pct"] > 50
    idx_ok  = out["index_linkage"]["linkable_via_indices"]
    out["summary"] = {
        "template_invertible_with_stored_seed": bool(inv_ok),
        "cross_matching_feasible_full_db":      bool(xm_ok),
        "linkable_via_reliable_indices":        bool(idx_ok),
        "recommendation": (
            "Si alguno de los tres indicadores es cierto, la proteccion no "
            "puede descansar en la transformacion cancelable bajo compromiso "
            "total de la base de datos. Mitigacion: derivar transform_seed de "
            "un secreto en poder del usuario (contrasena o token) mediante una "
            "KDF y no almacenarlo, o custodiarlo en un modulo separado; y "
            "derivar los indices fiables tambien del seed para que la "
            "revocacion los renueve."),
    }
    return out


# ─────────────────────────────────────────────
# 5. FIGURAS
# ─────────────────────────────────────────────

def plot_revocability(bers_same, bers_diff, bers_unrelated, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Revocability Evaluation — BER between Templates",
                 fontsize=13, fontweight='bold')

    ax1.hist(bers_same,     bins=40, alpha=0.7, color='#2196F3', density=True,
             label=f"Same user, same seed\nmu={np.mean(bers_same):.3f}")
    ax1.hist(bers_diff,     bins=40, alpha=0.7, color='#F44336', density=True,
             label=f"Same user, revoked seed\nmu={np.mean(bers_diff):.3f}")
    ax1.hist(bers_unrelated, bins=40, alpha=0.5, color='#9E9E9E', density=True,
             label=f"Different users\nmu={np.mean(bers_unrelated):.3f}")
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
                label="Random reference (0.5)")
    ax1.set_xlabel("BER (normalized Hamming)")
    ax1.set_ylabel("Density")
    ax1.set_title("BER distribution by comparison type")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    labels = ["Same seed", "Revoked seed", "Different users"]
    means  = [np.mean(bers_same), np.mean(bers_diff), np.mean(bers_unrelated)]
    stds   = [np.std(bers_same),  np.std(bers_diff),  np.std(bers_unrelated)]
    colors_bar = ['#2196F3', '#F44336', '#9E9E9E']
    bars = ax2.bar(labels, means, color=colors_bar, alpha=0.85,
                   yerr=stds, capsize=5)
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1.5,
                label="Reference 0.5")
    ax2.set_ylabel("Mean BER")
    ax2.set_title("Mean BER by comparison type")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_template_protection_revocability.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Revocability figure saved")
    return path


def plot_unlinkability(bers_related, bers_unrelated, results_dir):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Unlinkability Evaluation",
                 fontsize=13, fontweight='bold')

    ax.hist(bers_related,   bins=40, alpha=0.7, color='#F44336', density=True,
            label=f"Mated (same user, different seed)\nmu={np.mean(bers_related):.3f}")
    ax.hist(bers_unrelated, bins=40, alpha=0.7, color='#9E9E9E', density=True,
            label=f"Non-mated (different users)\nmu={np.mean(bers_unrelated):.3f}")
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
               label="Random reference (0.5)")
    ax.set_xlabel("BER between templates")
    ax.set_ylabel("Density")
    ax.set_title("BER distribution: mated vs non-mated")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Medida local D(s) del marco de Gomez-Barrero
    gb, centers, d_local = _ec.unlinkability_gomez_barrero(
        bers_related, bers_unrelated)
    ax2.plot(centers, d_local, color='#4a148c', linewidth=2)
    ax2.axhline(gb["D_sys_gomez_barrero"], color='#F44336', linestyle='--',
                label=f"D_sys = {gb['D_sys_gomez_barrero']:.4f}")
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("BER between templates")
    ax2.set_ylabel("D(s)")
    ax2.set_title("Local linkability measure (Gomez-Barrero 2018)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_template_protection_unlinkability.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Unlinkability figure saved")
    return path


# ─────────────────────────────────────────────
# 6. PDF
# ─────────────────────────────────────────────

def generate_pdf(run_name, noninv, revoc, unlink,
                 fig_rev, fig_unlink, results_dir,
                 attacks=None, dof=None, hw=None, seed=None,
                 n_bits=256, n_parity=16):
    pdf_path = str(Path(results_dir) / "eval_template_protection_report.pdf")
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=18,
                              textColor=colors.HexColor('#4a148c'))
    h1_s    = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13,
                              textColor=colors.HexColor('#4a148c'),
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

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#4a148c')),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#f3e5f5'), colors.white]),
            ('GRID',          (0,0),(-1,-1), 0.5, colors.lightgrey),
            ('PADDING',       (0,0),(-1,-1), 5),
            ('ALIGN',         (1,1),(-1,-1), 'CENTER'),
        ]))
        return t

    story = []
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Template Protection Evaluation", title_s))
    story.append(Paragraph("ISO/IEC 24745", styles['Heading2']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#4a148c')))
    story.append(Spacer(1, 0.5*cm))
    meta = [["Experiment", run_name],
            ["Date", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Template", f"{n_bits} bits"],
            ["RS parity", f"{n_parity} bytes (secret {(n_bits//8-n_parity)*8} bits)"],
            ["Threat model",
             "Full DB compromise (ISO/IEC 24745); no biometric sample"]]
    if seed is not None:
        meta.append(["RNG seed", str(seed)])
    if hw:
        meta.append(["Hardware", Paragraph(_ec.hardware_summary(hw), cell_wrap_s)])
    story.append(tbl(meta, [4.5*cm, 11.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    # 1. Non-invertibility
    story.append(Paragraph("1. Non-Invertibility", h1_s))
    story.append(Paragraph(
        "The commitment delta = xc XOR c combines the cancelable template "
        "with the encoded secret. To guarantee non-invertibility, the "
        "commitment must behave like random noise: a fraction of ones "
        "~0.5 and entropy ~1.0 bit/bit. Its correlation with the original "
        "template must be close to zero.", body_s))
    ni_data = [
        ["Metric", "Value", "Expected value"],
        ["Fraction of ones (mean)",
         f"{noninv['bit_fraction_mean']:.4f} +/- {noninv['bit_fraction_std']:.4f}",
         "~0.5"],
        ["Entropy (bits/bit)",
         f"{noninv['entropy_mean']:.4f} +/- {noninv['entropy_std']:.4f}",
         "~1.0"],
        ["Correlation with template",
         f"{noninv['correlation_mean']:.4f} +/- {noninv['correlation_std']:.4f}",
         "~0.0"],
        ["Subjects analyzed", str(noninv['n_users_analyzed']), "\u2014"],
    ]
    story.append(tbl(ni_data, [7*cm, 5*cm, 4*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>These three statistics are guaranteed by construction: since "
        "delta = xc XOR c and c derives from a random secret, the "
        "commitment is uniform regardless of the input biometric. They "
        "serve as a sanity check of the pipeline, not as evidence of "
        "non-invertibility. That evidence is in the effective entropy and "
        "in the Section 4 attacks.</i>", cap_s))
    story.append(Spacer(1, 0.3*cm))

    if dof:
        story.append(Paragraph(
            "<b>1.1 Effective entropy — degrees of freedom</b>", body_s))
        story.append(Paragraph(
            "Daugman's degrees of freedom estimate how many template bits "
            "are statistically independent, from the spread of the "
            "impostor distribution: DoF = p(1-p)/sigma^2. A DoF close to "
            "the nominal number of bits indicates that the transform does "
            "not introduce correlations that reduce the effective entropy.",
            body_s))
        dof_data = [
            ["Metric", "Value"],
            ["Impostor BER (mean +/- std)",
             f"{dof['impostor_ber_mean']:.4f} +/- {dof['impostor_ber_std']:.4f}"],
            ["Estimated degrees of freedom", f"{dof['degrees_of_freedom']:.1f}"],
            ["Nominal template bits", str(dof['n_bits_nominal'])],
            ["DoF / nominal-bits ratio", f"{dof['dof_ratio']}"],
            ["Comparisons analyzed", f"{dof['n_comparisons']:,}"],
        ]
        story.append(tbl(dof_data, [9*cm, 7*cm]))
        story.append(Spacer(1, 0.4*cm))

    # 2. Revocability
    story.append(Paragraph("2. Revocability", h1_s))
    story.append(Paragraph(
        "To assess revocability, two templates of the same user are "
        "generated with different seeds. The BER between revoked "
        "templates must be ~0.5, showing that there is no correlation "
        "between the old and the new template.",
        body_s))
    rv = revoc
    rev_data = [
        ["Comparison", "Mean BER", "BER std", "Interpretation"],
        ["Same user, same seed",
         f"{rv['same_user_same_seed']['ber_mean']:.4f}",
         f"{rv['same_user_same_seed']['ber_std']:.4f}",
         "Real intra-class variability (enroll vs test)"],
        ["Same user, revoked seed",
         f"{rv['same_user_diff_seed']['ber_mean']:.4f}",
         f"{rv['same_user_diff_seed']['ber_std']:.4f}",
         "Should be ~0.5"],
        ["Different users",
         f"{rv['diff_user']['ber_mean']:.4f}",
         f"{rv['diff_user']['ber_std']:.4f}",
         "Impostor reference"],
    ]
    story.append(tbl(rev_data, [5.5*cm, 3*cm, 2.5*cm, 5*cm]))
    story.append(Spacer(1, 0.3*cm))
    if Path(fig_rev).exists():
        story.append(RLImage(fig_rev, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 1. BER distribution for the three comparison types.",
            cap_s))

    # 3. Unlinkability
    story.append(PageBreak())
    story.append(Paragraph("3. Unlinkability", h1_s))
    story.append(Paragraph(
        "The BER distribution between templates of the same user with "
        "different seeds (mated) is compared against templates of "
        "different users (non-mated). For perfect unlinkability, both "
        "distributions must be statistically indistinguishable (D_sys ~ 0).",
        body_s))
    ul = unlink
    unlink_data = [
        ["Metric", "Value", "Interpretation"],
        ["Mated BER (same user, different seed)",
         f"{ul['related_templates']['ber_mean']:.4f} "
         f"(n={ul['related_templates']['n']:,})",
         "Should be ~0.5"],
        ["Non-mated BER (different users)",
         f"{ul['unrelated_templates']['ber_mean']:.4f} "
         f"(n={ul['unrelated_templates']['n']:,})",
         "Reference"],
        ["D_sys (Gomez-Barrero, TIFS 2018)",
         f"{ul['D_sys']:.4f}", "[0,1]; < 0.1 -> UNLINKABLE"],
        ["Maximum local D",
         f"{ul['gomez_barrero']['D_local_max']:.4f}", "Worst-case point"],
        ["Effect size (Cohen's d)",
         f"{ul['effect_size_cohens_d']:.4f}",
         "Magnitude reported as D_sys in previous versions"],
        ["Verdict", ul['verdict'], ""],
    ]
    story.append(tbl(unlink_data, [8*cm, 3.5*cm, 4.5*cm]))
    story.append(Spacer(1, 0.3*cm))
    if Path(fig_unlink).exists():
        story.append(RLImage(fig_unlink, width=14*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 2. Mated vs non-mated BER distribution.",
            cap_s))

    # ── 4. Attacks ────────────────────────────────────────────────────────
    if attacks:
        story.append(PageBreak())
        story.append(Paragraph(
            "4. Attacks Under Full Database Compromise", h1_s))
        story.append(Paragraph(
            "Non-invertibility is evaluated through template inversion and "
            "secret reconstruction attacks, and revocability through "
            "cross-matching experiments. The assumed threat model is the "
            "following: the adversary has the "
            "complete record T = (delta, h(s), reliable indices, "
            "transform_seed) and complete knowledge of the algorithm, but "
            "no valid biometric sample.", body_s))

        ti = attacks["template_inversion"]
        story.append(Paragraph("<b>4.1 Template inversion</b>", body_s))
        story.append(Paragraph(
            "The cancelable transform is recovered by probing, applying it "
            "to the zero vector and to the canonical basis, without "
            "knowledge of its internal implementation. Cost: "
            f"{ti['transform_probe_cost_calls']} calls.", body_s))
        story.append(tbl([
            ["Scenario", "Recovered template BER", "Expected"],
            ["With the transform_seed stored",
             f"{ti['ber_with_stored_seed']:.4f}", "0.0 if invertible"],
            ["With an incorrect transform_seed (control)",
             f"{ti['ber_with_wrong_seed']:.4f}", "~0.5"],
            ["Exact recovery",
             f"{ti['exact_recovery_rate_pct']:.1f}%", "\u2014"],
            ["Subjects analyzed", str(ti['n_users']), "\u2014"],
        ], [7*cm, 5*cm, 4*cm]))
        story.append(Spacer(1, 0.3*cm))

        sr = attacks["secret_reconstruction"]
        story.append(Paragraph(
            "<b>4.2 Secret reconstruction and structural leakage</b>", body_s))
        story.append(tbl([
            ["Metric", "Value"],
            ["Secret size", f"{sr['secret_bits']} bits"],
            ["Exhaustive search", f"{sr['brute_force_evaluations']} Argon2 evals"],
            ["Estimated time (1 core)",
             f"{sr['brute_force_years_single_core']} years"],
            ["Coset size of x compatible with delta", sr['coset_size']],
            ["Structural leakage over the template",
             f"{sr['structural_leakage_bits']} bits of {sr['template_bits']}"],
        ], [9*cm, 7*cm]))
        story.append(Paragraph(
            "<i>Exhaustive search of the secret is infeasible. The "
            "structural leakage, however, is inherent to fuzzy commitment: "
            "given delta, the template is confined to a coset of the "
            f"Reed-Solomon code, of size {sr['coset_size']}. It should be "
            "stated explicitly in the security analysis rather than "
            "omitted.</i>", cap_s))
        story.append(Spacer(1, 0.3*cm))

        xm = attacks["cross_matching"]
        story.append(Paragraph("<b>4.3 Cross-matching after revocation</b>", body_s))
        story.append(tbl([
            ["Attack", "Success (95% CI)", "Expected"],
            ["Naive: old xc against new delta",
             _ec.format_rate(xm['naive_cross_match'], 1), "0%"],
            ["Informed: invert with seed_A, reapply with seed_B",
             _ec.format_rate(xm['informed_cross_match'], 1), "0% if protected"],
        ], [8*cm, 5*cm, 3*cm]))
        story.append(Spacer(1, 0.3*cm))

        il = attacks["index_linkage"]
        story.append(Paragraph(
            "<b>4.4 Linkage via reliable indices</b>", body_s))
        story.append(Paragraph(
            "The reliable indices are selected by the magnitude of the "
            "mean embedding's components and do not depend on the "
            "transform_seed, but they are stored in the record. The "
            "Jaccard overlap between the index sets of different templates "
            "is measured.",
            body_s))
        il_rows = [["Scenario", "Mean Jaccard", "n"]]
        for k, lab in [("same_capture_revoked_seed",
                        "Revocation without a new capture"),
                       ("re_enrolment_new_captures",
                        "Re-enrollment with different captures"),
                       ("different_users", "Different users (baseline)")]:
            r = il[k]
            il_rows.append([
                lab,
                f"{r['jaccard_mean']:.4f}" if r['jaccard_mean'] is not None else "n/a",
                str(r['n'])])
        il_rows.append(["Theoretical baseline (random subsets)",
                        f"{il['jaccard_baseline_random']:.4f}", "\u2014"])
        story.append(tbl(il_rows, [8*cm, 4*cm, 4*cm]))
        story.append(Spacer(1, 0.3*cm))

        sm = attacks["summary"]
        story.append(Paragraph("<b>4.5 Attack summary</b>", body_s))
        story.append(tbl([
            ["Indicator", "Result"],
            ["Template invertible with the stored seed",
             "YES" if sm["template_invertible_with_stored_seed"] else "NO"],
            ["Cross-matching feasible with the full DB",
             "YES" if sm["cross_matching_feasible_full_db"] else "NO"],
            ["Linkable via reliable indices",
             "YES" if sm["linkable_via_reliable_indices"] else "NO"],
        ], [11*cm, 5*cm]))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("<i>" + sm["recommendation"] + "</i>", cap_s))

    # ── 5. Conclusions ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Conclusions", h1_s))

    ok, caveats = [], []

    if dof and dof.get("dof_ratio") and dof["dof_ratio"] > 0.85:
        ok.append(f"effective entropy preserved ({dof['degrees_of_freedom']:.0f} "
                  f"degrees of freedom over {dof['n_bits_nominal']} nominal bits)")
    if abs(rv['same_user_diff_seed']['ber_mean'] - 0.5) < 0.05:
        ok.append("revocability confirmed at the template level "
                  f"(BER = {rv['same_user_diff_seed']['ber_mean']:.4f} between "
                  "revoked templates)")
    if ul['verdict'] == "UNLINKABLE":
        ok.append(f"unlinkability of the bit vectors verified "
                  f"(D_sys = {ul['D_sys']:.4f} < 0.1)")

    if attacks:
        sm = attacks["summary"]
        if sm["template_invertible_with_stored_seed"]:
            caveats.append(
                "the cancelable transform is invertible given the "
                "transform_seed, which is currently stored alongside the "
                "template")
        if sm["cross_matching_feasible_full_db"]:
            caveats.append(
                "cross-matching between revoked templates is feasible for "
                "an adversary with access to the full database")
        if sm["linkable_via_reliable_indices"]:
            caveats.append(
                "the reliable indices are not renewed upon revocation and "
                "allow records of the same user to be linked")

    story.append(Paragraph(
        ("The system satisfies: " + "; ".join(ok) + ".") if ok
        else "Review the results individually.", body_s))

    if caveats:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "<b>Limitations identified under full database compromise:</b> "
            + "; ".join(caveats) + ".", body_s))
        story.append(Paragraph(
            "These limitations do not affect the cryptographic construction "
            "itself — the secret remains protected by Argon2 and exhaustive "
            "search remains infeasible — but rather where the transform "
            "material is kept. The proposed mitigation is to derive the "
            "transform_seed from a secret held by the user via a KDF "
            "without storing it, and to also derive the reliable indices "
            "from it so that revocation renews them.", body_s))

    doc.build(story)
    print(f"  \u2713 PDF saved to {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Evaluacion proteccion de templates")
    parser.add_argument("--run_name", type=str, default="exp08")
    parser.add_argument("--n_bits",   type=int, default=256)
    parser.add_argument("--n_parity", type=int, default=16)
    parser.add_argument("--n_users",  type=int, default=200)
    parser.add_argument("--seed",     type=int, default=_ec.DEFAULT_SEED,
                        help="RNG seed for reproducibility")
    parser.add_argument("--skip_attacks", action="store_true",
                        help="Skip the attack battery")
    args = parser.parse_args()

    rng = _ec.make_rng(args.seed)
    hw  = _ec.get_hardware_info()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)
    results_dir   = str(Path(config["paths"]["results"]) / args.run_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUACION PROTECCION DE TEMPLATES — {args.run_name}")
    print(f"{'='*60}")
    print(f"  Semilla={args.seed} | n_bits={args.n_bits} | n_parity={args.n_parity}")
    print(f"  {_ec.hardware_summary(hw)}")

    possible     = ["enroll_joint","enroll_phase2","enroll_phase1","enroll"]
    enroll_split = next((s for s in possible
                         if (Path(features_root)/s/"embeddings.npy").exists()), None)
    test_split   = enroll_split.replace("enroll", "test")
    enroll_embs, enroll_classes = load_split(features_root, enroll_split)
    test_embs,   test_classes   = load_split(features_root, test_split)

    print(f"\n  [1/5] No-invertibilidad (estadisticos del commitment)...")
    noninv = analyze_non_invertibility(enroll_embs, enroll_classes,
                                        args.n_bits, args.n_parity,
                                        min(args.n_users, 100), rng=rng)
    print(f"  Entropia media del commitment: {noninv['entropy_mean']:.4f} bits/bit")
    print(f"  Correlacion con template: {noninv['correlation_mean']:.4f}")

    print(f"\n  [2/5] Revocabilidad ({args.n_users} usuarios)...")
    revoc_stats, bers_same, bers_diff, bers_unrel = analyze_revocability(
        enroll_embs, enroll_classes,
        test_embs, test_classes,
        args.n_bits, args.n_users)
    print(f"  BER mismo seed: {revoc_stats['same_user_same_seed']['ber_mean']:.4f}")
    print(f"  BER seed revocado: {revoc_stats['same_user_diff_seed']['ber_mean']:.4f} (esperado ~0.5)")

    print(f"\n  [3/5] Unlinkability ({args.n_users} usuarios)...")
    unlink_stats, bers_rel, bers_unlink = analyze_unlinkability(
        enroll_embs, enroll_classes, args.n_bits, args.n_users, rng=rng)
    print(f"  D_sys (Gomez-Barrero) = {unlink_stats['D_sys']:.4f}"
          f"  → {unlink_stats['verdict']}")
    print(f"  Cohen's d = {unlink_stats['effect_size_cohens_d']:.4f}"
          f"  (lo que antes se reportaba como D_sys)")
    if unlink_stats["gomez_barrero"].get("sample_warning"):
        print(f"  ⚠ {unlink_stats['gomez_barrero']['sample_warning']}")

    # Entropia efectiva: la distribucion impostora de la fase de revocabilidad
    # es exactamente la que necesita el estimador de grados de libertad.
    print(f"\n  [4/5] Entropia efectiva (grados de libertad)...")
    dof = _ec.daugman_degrees_of_freedom(bers_unrel, args.n_bits)
    print(f"  DoF = {dof['degrees_of_freedom']:.1f} sobre "
          f"{dof['n_bits_nominal']} bits nominales "
          f"(ratio {dof['dof_ratio']})")

    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    attacks = None
    if not args.skip_attacks:
        print(f"\n  [5/5] Ataques bajo compromiso total de BD...")
        attacks = run_security_attacks(
            enroll_by_class, args.n_bits, args.n_parity, args.n_users, rng,
            emb_dim=int(enroll_embs.shape[1]))
        sm = attacks["summary"]
        print(f"  Template invertible con seed almacenado: "
              f"{'SI' if sm['template_invertible_with_stored_seed'] else 'NO'}")
        print(f"  Cross-matching viable con BD completa:   "
              f"{'SI' if sm['cross_matching_feasible_full_db'] else 'NO'}")
        print(f"  Enlazable via indices fiables:           "
              f"{'SI' if sm['linkable_via_reliable_indices'] else 'NO'}")

    # JSON
    results = {
        "run_name":           args.run_name,
        "date":               datetime.now().isoformat(),
        "seed":               args.seed,
        "hardware":           hw,
        "n_bits":             args.n_bits,
        "n_parity":           args.n_parity,
        "n_users_requested":  args.n_users,
        "non_invertibility":  noninv,
        "effective_entropy":  dof,
        "revocability":       revoc_stats,
        "unlinkability":      unlink_stats,
        "security_attacks":   attacks,
    }
    json_path = _ec.save_json(
        results, Path(results_dir) / "eval_template_protection_results.json")
    print(f"\n  ✓ JSON guardado")

    print("\n  Generando figuras...")
    fig_rev    = plot_revocability(bers_same, bers_diff, bers_unrel, results_dir)
    fig_unlink = plot_unlinkability(bers_rel, bers_unlink, results_dir)

    print("\n  Generando PDF...")
    generate_pdf(args.run_name, noninv, revoc_stats, unlink_stats,
                 fig_rev, fig_unlink, results_dir,
                 attacks=attacks, dof=dof, hw=hw, seed=args.seed,
                 n_bits=args.n_bits, n_parity=args.n_parity)

    print(f"\n{'='*60}")
    print(f"  COMPLETADO")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
