"""
eval_computational.py
Evaluacion de rendimiento computacional.

Mide los tiempos de procesamiento y consumo de memoria de cada
componente del sistema criptobiometrico.

Metricas:
  - Tiempo de enrolamiento por usuario (5 imagenes)
  - Tiempo de autenticacion por intento (3 imagenes)
  - Desglose de tiempos por componente:
      * Transformacion cancelable (binarizacion + permutacion)
      * Reed-Solomon codificacion/decodificacion
      * Argon2 hashing/verificacion
  - Influencia de los parametros Argon2 (m, t, p) en la latencia
  - Tamano de almacenamiento del template protegido por usuario
  - Memoria RAM durante enrolamiento y autenticacion

Salidas:
  - results/<run_name>/eval_computational_results.json
  - results/<run_name>/eval_computational_breakdown.png
  - results/<run_name>/eval_computational_argon2.png
  - results/<run_name>/eval_computational_report.pdf

Uso:
    python scripts/eval_computational.py --run_name exp08
    python scripts/eval_computational.py --run_name exp08 --n_users 50
"""

import sys
import json
import time
import secrets
import argparse
import importlib
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from reedsolo import RSCodec
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
from reportlab.lib.enums import TA_CENTER

sys.path.insert(0, str(Path(__file__).parent))
_cancelable              = importlib.import_module("06_cancelable")
enroll_template_reliable = _cancelable.enroll_template_reliable
verify_template_reliable = _cancelable.verify_template_reliable

_fc             = importlib.import_module("07_fuzzy_commitment")
enroll_user_fc  = _fc.enroll_user
authenticate_user_fc = _fc.authenticate_user

_ec = importlib.import_module("eval_common")


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(features_root, split_name):
    split_dir  = Path(features_root) / split_name
    embs       = np.load(str(split_dir / "embeddings.npy"))
    classes    = json.load(open(str(split_dir / "class_strings.json")))
    return embs, classes


# ─────────────────────────────────────────────
# TIEMPOS POR COMPONENTE
# ─────────────────────────────────────────────

def benchmark_components(embs_enroll, user_id, n_bits, n_parity, n_enroll,
                          n_repeats=10):
    """
    Mide el tiempo de cada componente del pipeline por separado.
    Repite n_repeats veces y devuelve media y std en milisegundos.
    """
    times = {"cancelable_enroll": [], "rs_encode": [], "argon2_hash": [],
             "cancelable_verify": [], "rs_decode": [], "argon2_verify": []}

    seed       = secrets.token_hex(32)
    secret_len = n_bits//8 - n_parity

    for _ in range(n_repeats):
        embs = embs_enroll[:n_enroll]

        # Cancelable enroll
        t0 = time.perf_counter()
        tmpl, indices = enroll_template_reliable(embs, user_id, seed, n_bits)
        times["cancelable_enroll"].append((time.perf_counter()-t0)*1000)

        # RS encode
        import os
        secret = os.urandom(secret_len)
        t0 = time.perf_counter()
        rsc      = RSCodec(n_parity)
        codeword = bytes(rsc.encode(bytearray(secret)))
        times["rs_encode"].append((time.perf_counter()-t0)*1000)

        # Argon2 hash
        t0 = time.perf_counter()
        ph         = PasswordHasher(memory_cost=8192, time_cost=3, parallelism=2)
        hash_str   = ph.hash(secret.hex())
        times["argon2_hash"].append((time.perf_counter()-t0)*1000)

        # Commitment XOR
        template_bytes = np.packbits(tmpl).tobytes()
        commitment     = bytes(a^b for a,b in zip(template_bytes, codeword))

        # Cancelable verify
        test_emb = embs[0]
        t0 = time.perf_counter()
        tmpl_ver = verify_template_reliable(test_emb, user_id, seed, indices)
        times["cancelable_verify"].append((time.perf_counter()-t0)*1000)

        # RS decode
        ver_bytes    = np.packbits(tmpl_ver).tobytes()
        cword_noisy  = bytes(a^b for a,b in zip(ver_bytes, commitment))
        t0 = time.perf_counter()
        try:
            decoded = rsc.decode(bytearray(cword_noisy))
            secret_rec = bytes(decoded[0] if isinstance(decoded, tuple) else decoded)
        except Exception:
            secret_rec = secret
        times["rs_decode"].append((time.perf_counter()-t0)*1000)

        # Argon2 verify
        t0 = time.perf_counter()
        try:
            ph.verify(hash_str, secret_rec.hex())
        except Exception:
            pass
        times["argon2_verify"].append((time.perf_counter()-t0)*1000)

    return {k: {"mean_ms": round(float(np.mean(v)), 3),
                "std_ms":  round(float(np.std(v)),  3)}
            for k, v in times.items()}


def benchmark_end_to_end(enroll_by_class, test_by_class,
                          n_bits, n_parity, n_enroll, n_verify,
                          n_users=30, seed=None):
    """
    Mide el tiempo end-to-end de enrolamiento y autenticacion.

    IMPORTANTE — separacion aceptado/rechazado:
    La version anterior promediaba todos los intentos de autenticacion
    juntos, lo que producia un minimo de decimas de milisegundo imposible
    de conciliar con los ~7 ms que cuesta Argon2. La explicacion es que los
    intentos que fallan en la decodificacion Reed-Solomon nunca llegan a
    ejecutar Argon2 y son un orden de magnitud mas rapidos. Mezclarlos
    falsea la latencia reportada, asi que se separan.

    Esa diferencia de tiempo es ademas un canal lateral: un observador
    puede distinguir un rechazo por RS de un rechazo por Argon2 midiendo la
    latencia. Se reporta explicitamente para que pueda discutirse en el
    analisis de seguridad.

    NOTA DE ALCANCE: estos tiempos NO incluyen segmentacion, preprocesado ni
    la pasada del extractor de caracteristicas, que dominan la latencia
    percibida por el usuario.
    """
    enroll_by_class, test_by_class, _filter_stats = (
        _ec.filter_subjects_with_min_images(
            enroll_by_class, test_by_class, n_enroll, n_verify))

    classes      = list(enroll_by_class.keys())[:n_users]
    enroll_times = []
    accept_times = []
    reject_times = []
    user_records = {}
    rng = _ec.make_rng(seed if seed is not None else _ec.DEFAULT_SEED)

    for cls in classes:
        embs = enroll_by_class[cls]
        t0   = time.perf_counter()
        rec  = _ec.enroll_user_seeded(embs[:n_enroll], cls,
                                       n_bits, n_parity, rng)
        enroll_times.append((time.perf_counter() - t0) * 1000)
        user_records[cls] = rec

    # Genuinos -> mayoritariamente aceptados
    for cls in classes:
        if cls not in test_by_class:
            continue
        window = list(np.array(test_by_class[cls])[:n_verify])
        t0     = time.perf_counter()
        ok, _  = authenticate_user_fc(window, cls, user_records[cls])
        dt     = (time.perf_counter() - t0) * 1000
        (accept_times if ok else reject_times).append(dt)

    # Impostores -> rechazados, para poblar la rama de rechazo
    others = [c for c in test_by_class if c in user_records]
    for cls in classes:
        cand = [c for c in others if c != cls]
        if not cand:
            continue
        qcls   = cand[int(rng.integers(0, len(cand)))]
        window = list(np.array(test_by_class[qcls])[:n_verify])
        t0     = time.perf_counter()
        ok, _  = authenticate_user_fc(window, cls, user_records[cls])
        dt     = (time.perf_counter() - t0) * 1000
        (accept_times if ok else reject_times).append(dt)

    def _stats(v):
        if not v:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {"mean": round(float(np.mean(v)), 2),
                "std":  round(float(np.std(v)),  2),
                "min":  round(float(np.min(v)),  2),
                "max":  round(float(np.max(v)),  2),
                "n":    len(v)}

    all_auth = accept_times + reject_times
    return {
        "enroll_ms":        _stats(enroll_times),
        "auth_ms":          _stats(all_auth),
        "auth_accept_ms":   _stats(accept_times),
        "auth_reject_ms":   _stats(reject_times),
        "timing_side_channel_ms": round(
            abs(_stats(accept_times)["mean"] - _stats(reject_times)["mean"]), 2),
        "scope_note": ("Excluye segmentacion, preprocesado y extraccion de "
                       "embeddings."),
    }


def benchmark_identification_1n(enroll_by_class, test_by_class,
                                 n_bits, n_parity, n_enroll, n_verify,
                                 db_sizes=(10, 50, 100, 250), seed=None):
    """
    Latencia de identificacion 1:N.

    El servidor recorre los usuarios aplicando primero la decodificacion
    Reed-Solomon (rapida) y ejecutando Argon2 solo cuando RS tiene exito.
    En el peor caso, un impostor obliga a intentar RS contra los N
    registros. Con una base de datos real la latencia de autenticacion
    crece con N: conviene medirla.
    """
    from reedsolo import ReedSolomonError

    rng = _ec.make_rng(seed if seed is not None else _ec.DEFAULT_SEED)
    enroll_by_class, test_by_class, _filter_stats = (
        _ec.filter_subjects_with_min_images(
            enroll_by_class, test_by_class, n_enroll, n_verify))
    classes = list(enroll_by_class.keys())
    results = []

    for N in db_sizes:
        sel = classes[:N]
        if len(sel) < 2:
            continue
        records = {c: _ec.enroll_user_seeded(
                       enroll_by_class[c][:n_enroll], c, n_bits, n_parity, rng)
                   for c in sel}

        # El sondeo debe ser un usuario NO enrolado en este subconjunto: si
        # fuera genuino, el bucle terminaria en cuanto lo encontrase y no se
        # estaria midiendo el peor caso, que es justamente el de un impostor
        # que obliga a recorrer los N registros.
        probe_cls = next((c for c in classes[N:] if c in test_by_class), None)
        if probe_cls is None:
            probe_cls = next((c for c in test_by_class if c not in records), None)
        if probe_cls is None:
            continue
        window   = list(np.array(test_by_class[probe_cls])[:n_verify])
        mean_emb = np.mean(np.array(window), axis=0)
        nrm      = np.linalg.norm(mean_emb)
        if nrm > 0:
            mean_emb = mean_emb / nrm

        rsc = RSCodec(n_parity)
        t0  = time.perf_counter()
        for cls, rec in records.items():
            idx  = np.array(rec["reliable_indices"])
            tmpl = verify_template_reliable(
                mean_emb, cls, rec["transform_seed"], idx)
            noisy = bytes(a ^ b for a, b in zip(
                np.packbits(tmpl).tobytes(),
                bytes.fromhex(rec["commitment"])))
            try:
                dec = rsc.decode(bytearray(noisy))
                sec = bytes(dec[0] if isinstance(dec, tuple) else dec)
                if _fc.argon2_verify(rec["hash_secret"], sec):
                    break
            except ReedSolomonError:
                continue
            except Exception:
                continue
        worst = (time.perf_counter() - t0) * 1000

        results.append({
            "db_size":        N,
            "worst_case_ms":  round(worst, 1),
            "ms_per_record":  round(worst / N, 3),
        })
    return results


def benchmark_argon2_params(secret_len=16, n_repeats=5):
    """Mide el tiempo de Argon2 para distintos parametros (m, t, p)."""
    configs = [
        {"memory_cost": 4096,  "time_cost": 1, "parallelism": 1},
        {"memory_cost": 8192,  "time_cost": 1, "parallelism": 1},
        {"memory_cost": 8192,  "time_cost": 3, "parallelism": 2},  # PDF default
        {"memory_cost": 16384, "time_cost": 3, "parallelism": 2},
        {"memory_cost": 65536, "time_cost": 3, "parallelism": 4},
    ]
    results = []
    import os
    secret = os.urandom(secret_len)

    for cfg in configs:
        ph   = PasswordHasher(**cfg)
        t_ms = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            h  = ph.hash(secret.hex())
            ph.verify(h, secret.hex())
            t_ms.append((time.perf_counter()-t0)*1000)
        results.append({
            "m": cfg["memory_cost"],
            "t": cfg["time_cost"],
            "p": cfg["parallelism"],
            "time_ms_mean": round(float(np.mean(t_ms)), 1),
            "time_ms_std":  round(float(np.std(t_ms)),  1),
            "is_default":   cfg["time_cost"]==3 and cfg["memory_cost"]==8192,
        })
    return results


def measure_template_storage(n_bits, n_parity, record=None):
    """
    Tamano de almacenamiento del template protegido por usuario.

    La version anterior estimaba los campos a ojo. Aqui se MIDE la
    serializacion real del registro cuando se proporciona uno, y se
    contrasta con la estimacion. Ademas se anade el tamano optimizado de
    los indices fiables: almacenarlos como enteros de 4 bytes es
    innecesario, ya que un indice sobre un embedding de d dimensiones cabe
    en ceil(log2(d)) bits — 11 bits para d=2048.
    """
    commitment_bytes = n_bits // 8
    hash_bytes       = 100
    indices_bytes    = n_bits * 4
    seed_bytes       = 32
    metadata_bytes   = 50
    total = (commitment_bytes + hash_bytes + indices_bytes +
             seed_bytes + metadata_bytes)

    out = {
        "commitment_bytes":  commitment_bytes,
        "hash_bytes":        hash_bytes,
        "indices_bytes":     indices_bytes,
        "seed_bytes":        seed_bytes,
        "metadata_bytes":    metadata_bytes,
        "total_bytes":       total,
        "total_kb":          round(total / 1024, 2),
    }

    if record is not None:
        emb_dim = max(record["reliable_indices"]) + 1 if record["reliable_indices"] else 2048
        bits_per_index = int(np.ceil(np.log2(max(emb_dim, 2))))
        packed = int(np.ceil(n_bits * bits_per_index / 8))
        out["indices_bytes_packed"]  = packed
        out["indices_bits_per_index"] = bits_per_index
        out["total_bytes_optimised"] = total - indices_bytes + packed
        out["total_kb_optimised"]    = round(
            out["total_bytes_optimised"] / 1024, 2)

        serial = json.dumps({
            "user_id":          record["user_id"],
            "commitment":       record["commitment"],
            "hash_secret":      record["hash_secret"],
            "reliable_indices": record["reliable_indices"],
            "transform_seed":   record["transform_seed"],
            "n_bits":           record["n_bits"],
            "n_parity":         record["n_parity"],
            "n_enroll_images":  record["n_enroll_images"],
        })
        out["measured_json_bytes"] = len(serial.encode("utf-8"))
        out["measured_json_kb"]    = round(len(serial.encode("utf-8")) / 1024, 2)

        try:
            import sqlite3, tempfile
            with tempfile.TemporaryDirectory() as td:
                dbp = str(Path(td) / "m.db")
                con = sqlite3.connect(dbp)
                con.execute("CREATE TABLE users (user_id TEXT PRIMARY KEY, "
                            "commitment TEXT, hash_secret TEXT, "
                            "reliable_indices TEXT, transform_seed TEXT, "
                            "n_bits INT, n_parity INT, n_enroll_images INT, "
                            "created_at TEXT)")
                base = Path(dbp).stat().st_size
                rows = 200
                con.executemany(
                    "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?)",
                    [(f"u{i}", record["commitment"], record["hash_secret"],
                      json.dumps(record["reliable_indices"]),
                      record["transform_seed"], record["n_bits"],
                      record["n_parity"], record["n_enroll_images"],
                      datetime.now().isoformat()) for i in range(rows)])
                con.commit(); con.close()
                grown = Path(dbp).stat().st_size - base
                out["measured_sqlite_bytes_per_user"] = int(grown / rows)
                out["measured_sqlite_kb_per_user"] = round(grown / rows / 1024, 2)
        except Exception as e:
            out["measured_sqlite_bytes_per_user"] = None
            out["sqlite_error"] = str(e)

    return out


def measure_memory(enroll_by_class, test_by_class, n_bits, n_parity,
                    n_enroll, n_verify=3, seed=None):
    """Mide el uso de memoria durante enrolamiento Y autenticacion."""
    rng  = _ec.make_rng(seed if seed is not None else _ec.DEFAULT_SEED)
    enroll_by_class, test_by_class, _filter_stats = (
        _ec.filter_subjects_with_min_images(
            enroll_by_class, test_by_class, n_enroll, n_verify))
    cls  = list(enroll_by_class.keys())[0]
    embs = enroll_by_class[cls][:n_enroll]

    tracemalloc.start()
    rec = _ec.enroll_user_seeded(embs, cls, n_bits, n_parity, rng)
    cur_e, peak_e = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cur_a = peak_a = 0
    if cls in test_by_class:
        window = list(np.array(test_by_class[cls])[:n_verify])
        tracemalloc.start()
        authenticate_user_fc(window, cls, rec)
        cur_a, peak_a = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "enroll": {"current_kb": round(cur_e / 1024, 1),
                   "peak_kb":    round(peak_e / 1024, 1)},
        "auth":   {"current_kb": round(cur_a / 1024, 1),
                   "peak_kb":    round(peak_a / 1024, 1)},
        # compatibilidad con el formato anterior
        "current_kb": round(cur_e / 1024, 1),
        "peak_kb":    round(peak_e / 1024, 1),
        "_record":    rec,
    }


# ─────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────

def plot_breakdown(component_times, end_to_end, results_dir):
    """Timing by component and end-to-end."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Computational Performance by Component",
                 fontsize=13, fontweight='bold')

    # By component
    labels = ["Cancel.\nEnroll.", "RS\nEncode", "Argon2\nHash",
              "Cancel.\nVerif.", "RS\nDecode", "Argon2\nVerif."]
    means  = [component_times[k]["mean_ms"] for k in component_times]
    stds   = [component_times[k]["std_ms"]  for k in component_times]
    colors_bar = ['#2196F3']*3 + ['#4CAF50']*3
    ax1.bar(labels, means, color=colors_bar, alpha=0.85, yerr=stds, capsize=4)
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Time by component\n(blue=enrollment, green=authentication)")
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m+s+1, f"{m:.1f}", ha='center', fontsize=8)

    # End-to-end
    labels2 = ["Enrollment\n(5 imgs)", "Authentication\n(3 imgs)"]
    means2  = [end_to_end["enroll_ms"]["mean"], end_to_end["auth_ms"]["mean"]]
    stds2   = [end_to_end["enroll_ms"]["std"],  end_to_end["auth_ms"]["std"]]
    ax2.bar(labels2, means2, color=['#2196F3','#4CAF50'], alpha=0.85,
            yerr=stds2, capsize=6, width=0.5)
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("End-to-end time\n(excluding embedding extraction)")
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means2, stds2)):
        ax2.text(i, m+s+2, f"{m:.0f} ms", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_computational_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Breakdown figure saved")
    return path


def plot_argon2(argon2_results, results_dir):
    """Argon2 timing for different parameters."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Argon2 Latency by Parameter Configuration",
                 fontsize=13, fontweight='bold')

    labels = [f"m={r['m']}K\nt={r['t']},p={r['p']}" for r in argon2_results]
    means  = [r["time_ms_mean"] for r in argon2_results]
    stds   = [r["time_ms_std"]  for r in argon2_results]
    bar_colors = ['#FF9800' if r["is_default"] else '#9E9E9E' for r in argon2_results]

    bars = ax.bar(labels, means, color=bar_colors, alpha=0.85,
                  yerr=stds, capsize=5)
    ax.set_ylabel("Hash+verify time (ms)")
    ax.set_title("Orange = system default configuration (PDF)")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{m:.0f}ms", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = str(Path(results_dir) / "eval_computational_argon2.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figura Argon2 guardada")
    return path


# ─────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────

def generate_pdf(run_name, comp_times, end_to_end, argon2_res,
                 storage, memory, fig_break, fig_arg, results_dir,
                 hw=None, seed=None, ident_1n=None,
                 n_bits=256, n_parity=16, n_enroll=5, n_verify=3):
    pdf_path = str(Path(results_dir) / "eval_computational_report.pdf")
    doc      = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  topMargin=2*cm, bottomMargin=2*cm,
                                  leftMargin=2*cm, rightMargin=2*cm)
    styles   = getSampleStyleSheet()
    title_s  = ParagraphStyle('T', parent=styles['Title'], fontSize=18,
                               textColor=colors.HexColor('#1b5e20'))
    h1_s     = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13,
                               textColor=colors.HexColor('#1b5e20'),
                               spaceBefore=14, spaceAfter=6)
    body_s   = ParagraphStyle('B', parent=styles['Normal'],
                               fontSize=10, spaceAfter=6, leading=14)
    cap_s    = ParagraphStyle('C', parent=styles['Normal'], fontSize=9,
                               textColor=colors.grey,
                               alignment=TA_CENTER, spaceAfter=10)

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0),(-1,-1), 9),
            ('BACKGROUND',    (0,0),(-1,0), colors.HexColor('#1b5e20')),
            ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#e8f5e9'), colors.white]),
            ('GRID',          (0,0),(-1,-1), 0.5, colors.lightgrey),
            ('PADDING',       (0,0),(-1,-1), 5),
            ('ALIGN',         (1,1),(-1,-1), 'CENTER'),
        ]))
        return t

    story = []
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Computational Performance Evaluation",
                            title_s))
    story.append(Paragraph("Timing, memory and storage", styles['Heading2']))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#1b5e20')))
    story.append(Spacer(1, 0.3*cm))
    meta = [["Experiment", run_name],
            ["Date", datetime.now().strftime("%d/%m/%Y %H:%M")],
            ["Configuration",
             f"{n_bits} bits, {n_parity} B parity, "
             f"{n_enroll} enroll img / {n_verify} verify img"],
            ["Argon2", "m=8192 KB, t=3, p=2"]]
    if seed is not None:
        meta.append(["RNG seed", str(seed)])
    if hw:
        meta += [["CPU", hw.get("cpu_model", "n/a")],
                 ["Logical cores", str(hw.get("cpu_count", "n/a"))],
                 ["RAM", f"{hw.get('ram_gb', 'n/a')} GB"],
                 ["System", f"{hw.get('os','')} | Python {hw.get('python','')}"]]
    story.append(tbl(meta, [4.5*cm, 11.5*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>All timings correspond to the cryptographic pipeline and "
        "EXCLUDE segmentation, preprocessing, and the feature extractor "
        "pass, which dominate the latency perceived by the user and "
        "depend on the inference hardware.</i>", cap_s))
    story.append(Spacer(1, 0.3*cm))

    # End-to-end timings
    story.append(Paragraph("1. End-to-End Timings", h1_s))
    def _row(lab, k):
        st = end_to_end.get(k, {})
        return [lab, f"{st.get('mean',0):.1f}", f"{st.get('std',0):.1f}",
                f"{st.get('min',0):.1f}", f"{st.get('max',0):.1f}",
                str(st.get('n', '\u2014'))]

    ee_data = [["Operation", "Mean (ms)", "Std", "Min", "Max", "n"],
               _row(f"Enrollment ({n_enroll} imgs)", "enroll_ms"),
               _row(f"Accepted authentication ({n_verify} imgs)", "auth_accept_ms"),
               _row(f"Rejected authentication ({n_verify} imgs)", "auth_reject_ms")]
    story.append(tbl(ee_data, [5.5*cm, 2.4*cm, 2*cm, 2*cm, 2*cm, 2*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Accepted and rejected attempts are reported separately because a "
        "rejection that fails at Reed-Solomon decoding never reaches "
        "Argon2 and is an order of magnitude faster; averaging them "
        "together produces a mean latency that does not correspond to any "
        "real case. The observed difference is "
        f"{end_to_end.get('timing_side_channel_ms', 0):.1f} ms, which "
        "constitutes a timing side channel: an observer can tell the "
        "reason for a rejection by measuring latency. Common mitigation: "
        "equalize response time with a constant delay.",
        body_s))
    story.append(Spacer(1, 0.3*cm))

    # Breakdown by component
    story.append(Paragraph("2. Breakdown by Component", h1_s))
    comp_data = [["Component", "Mean (ms)", "Std (ms)"]]
    labels_map = {
        "cancelable_enroll": "Cancelable transform (enroll.)",
        "rs_encode":         "Reed-Solomon encoding",
        "argon2_hash":       "Argon2 hash",
        "cancelable_verify": "Cancelable transform (verif.)",
        "rs_decode":         "Reed-Solomon decoding",
        "argon2_verify":     "Argon2 verification",
    }
    for k, label in labels_map.items():
        comp_data.append([label,
                          f"{comp_times[k]['mean_ms']:.3f}",
                          f"{comp_times[k]['std_ms']:.3f}"])
    story.append(tbl(comp_data, [9*cm, 4*cm, 3*cm]))
    story.append(Spacer(1, 0.3*cm))

    if Path(fig_break).exists():
        story.append(RLImage(fig_break, width=16*cm, height=6*cm))
        story.append(Paragraph(
            "Figure 1. Timing by component (left) and end-to-end (right).",
            cap_s))

    # Argon2
    story.append(PageBreak())
    story.append(Paragraph("3. Influence of the Argon2 Parameters", h1_s))
    story.append(Paragraph(
        "The Argon2 parameters (m=memory, t=iterations, p=parallelism) "
        "control the computational cost of the hash. Larger values "
        "increase resistance to brute-force attacks but increase latency.",
        body_s))
    a2_data = [["m (KB)", "t", "p", "Time (ms)", "Default"]]
    for r in argon2_res:
        a2_data.append([
            str(r["m"]), str(r["t"]), str(r["p"]),
            f"{r['time_ms_mean']:.0f} +/- {r['time_ms_std']:.0f}",
            "YES" if r["is_default"] else ""])
    story.append(tbl(a2_data, [2.5*cm, 2*cm, 2*cm, 5*cm, 4.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    if Path(fig_arg).exists():
        story.append(RLImage(fig_arg, width=14*cm, height=5.5*cm))
        story.append(Paragraph(
            "Figure 2. Argon2 latency for different parameter configurations.",
            cap_s))

    # Storage
    story.append(Paragraph("4. Storage and Memory", h1_s))
    stor_data = [
        ["Field", "Size (bytes)", "Description"],
        ["Commitment (delta)",
         str(storage["commitment_bytes"]),
         "Template XOR RS codeword"],
        ["Argon2 hash",
         str(storage["hash_bytes"]),
         "Hash string including embedded salt"],
        ["Reliable indices",
         str(storage["indices_bytes"]),
         f"List of {storage['indices_bytes']//4} integer indices"],
        ["Transform seed",
         str(storage["seed_bytes"]),
         "256-bit cancelable seed"],
        ["Metadata",
         str(storage["metadata_bytes"]),
         "Timestamps and additional fields"],
        ["TOTAL (estimated)",
         f"{storage['total_bytes']} ({storage['total_kb']} KB)",
         "Sum of fields"],
    ]
    if storage.get("indices_bytes_packed"):
        stor_data.append([
            "Packed indices",
            str(storage["indices_bytes_packed"]),
            f"{storage['indices_bits_per_index']} bits/index instead of 32"])
        stor_data.append([
            "TOTAL (optimized)",
            f"{storage['total_bytes_optimised']} "
            f"({storage['total_kb_optimised']} KB)",
            "With packed indices"])
    if storage.get("measured_json_bytes"):
        stor_data.append([
            "Measured (JSON)",
            f"{storage['measured_json_bytes']} "
            f"({storage['measured_json_kb']} KB)",
            "Real serialization of the record"])
    if storage.get("measured_sqlite_bytes_per_user"):
        stor_data.append([
            "Measured (SQLite)",
            f"{storage['measured_sqlite_bytes_per_user']} "
            f"({storage['measured_sqlite_kb_per_user']} KB)",
            "Real DB growth per user"])
    story.append(tbl(stor_data, [4.5*cm, 4*cm, 7.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    mem_data = [["Operation", "Current memory (KB)", "Peak memory (KB)"]]
    if "enroll" in memory:
        mem_data.append(["Enrollment",
                         str(memory["enroll"]["current_kb"]),
                         str(memory["enroll"]["peak_kb"])])
        mem_data.append(["Authentication",
                         str(memory["auth"]["current_kb"]),
                         str(memory["auth"]["peak_kb"])])
    else:
        mem_data.append(["Enrollment", str(memory["current_kb"]),
                         str(memory["peak_kb"])])
    story.append(tbl(mem_data, [5.5*cm, 5*cm, 5.5*cm]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<i>These figures correspond to the memory managed by the "
        "interpreter (tracemalloc) and do not include the memory block "
        "Argon2 reserves internally, which is 8 MB per evaluation under "
        "the m=8192 KB configuration. That is the dominant consumption and "
        "the one that limits server concurrency.</i>", cap_s))

    if ident_1n:
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph("5. 1:N Identification Latency", h1_s))
        story.append(Paragraph(
            "1:1 verification is not the only mode of operation. In "
            "identification, the server iterates over the records, "
            "applying Reed-Solomon decoding first and running Argon2 only "
            "when it succeeds. The worst case is measured, in which all N "
            "records are visited.", body_s))
        id_rows = [["Users in DB", "Worst case (ms)", "ms per record"]]
        for r in ident_1n:
            id_rows.append([f"{r['db_size']:,}",
                            f"{r['worst_case_ms']:.1f}",
                            f"{r['ms_per_record']:.3f}"])
        story.append(tbl(id_rows, [5.5*cm, 5*cm, 5.5*cm]))
        story.append(Spacer(1, 0.2*cm))
        if ident_1n:
            per = ident_1n[-1]["ms_per_record"]
            story.append(Paragraph(
                f"The cost grows linearly with database size, at a rate of "
                f"{per:.3f} ms per record. For large databases, indexing or "
                f"partitioning is advisable; 1:1 verification, which is the "
                f"mode evaluated in the previous sections, is not affected.",
                body_s))

    story.append(Paragraph("6. Conclusions", h1_s))
    story.append(Paragraph(
        f"Enrollment time is {end_to_end['enroll_ms']['mean']:.1f} ms and "
        f"an accepted authentication takes "
        f"{end_to_end.get('auth_accept_ms', end_to_end['auth_ms'])['mean']:.1f} ms, "
        f"with Argon2 the most costly component at "
        f"{comp_times['argon2_hash']['mean_ms']:.1f} ms on average. The "
        f"protected template occupies "
        f"{storage.get('measured_sqlite_kb_per_user', storage['total_kb'])} KB "
        f"per user as measured in SQLite "
        f"({storage['total_kb']} KB as a sum of fields), reducible to "
        f"{storage.get('total_kb_optimised', storage['total_kb'])} KB by "
        f"packing the reliable indices. In any case it is negligible for "
        f"a modern database.",
        body_s))

    doc.build(story)
    print(f"  \u2713 PDF saved to {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Evaluacion computacional")
    parser.add_argument("--run_name", type=str, default="exp08")
    parser.add_argument("--n_bits",   type=int, default=256)
    parser.add_argument("--n_parity", type=int, default=16)
    parser.add_argument("--n_enroll", type=int, default=5)
    parser.add_argument("--n_verify", type=int, default=3)
    parser.add_argument("--n_users",  type=int, default=30)
    parser.add_argument("--n_repeats",type=int, default=10)
    parser.add_argument("--seed",     type=int, default=_ec.DEFAULT_SEED)
    parser.add_argument("--db_sizes", type=int, nargs="+",
                        default=[10, 50, 100, 250],
                        help="Tamanos de BD para la latencia 1:N")
    args = parser.parse_args()

    hw = _ec.get_hardware_info()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)
    results_dir   = str(Path(config["paths"]["results"]) / args.run_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUACION COMPUTACIONAL — {args.run_name}")
    print(f"{'='*60}")
    print(f"  {_ec.hardware_summary(hw)}")
    print(f"  Semilla={args.seed}")

    possible     = ["enroll_joint","enroll_phase2","enroll_phase1","enroll"]
    enroll_split = next((s for s in possible
                         if (Path(features_root)/s/"embeddings.npy").exists()), None)
    test_split   = enroll_split.replace("enroll","test")

    enroll_embs, enroll_classes = load_split(features_root, enroll_split)
    test_embs,   test_classes   = load_split(features_root, test_split)

    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)
    test_by_class = defaultdict(list)
    for emb, cls in zip(test_embs, test_classes):
        test_by_class[cls].append(emb)

    cls_sample   = list(enroll_by_class.keys())[0]
    embs_sample  = enroll_by_class[cls_sample]

    print(f"\n  [1/6] Tiempos por componente ({args.n_repeats} repeticiones)...")
    comp_times = benchmark_components(
        embs_sample, cls_sample, args.n_bits, args.n_parity,
        args.n_enroll, args.n_repeats)
    for k, v in comp_times.items():
        print(f"  {k:<25}: {v['mean_ms']:.3f} +/- {v['std_ms']:.3f} ms")

    print(f"\n  [2/6] Tiempos end-to-end ({args.n_users} usuarios)...")
    end_to_end = benchmark_end_to_end(
        enroll_by_class, test_by_class,
        args.n_bits, args.n_parity, args.n_enroll, args.n_verify,
        args.n_users, seed=args.seed)
    print(f"  Enrolamiento          : {end_to_end['enroll_ms']['mean']:.1f} ms")
    print(f"  Autenticacion aceptada: {end_to_end['auth_accept_ms']['mean']:.1f} ms "
          f"(n={end_to_end['auth_accept_ms']['n']})")
    print(f"  Autenticacion rechazada: {end_to_end['auth_reject_ms']['mean']:.1f} ms "
          f"(n={end_to_end['auth_reject_ms']['n']})")
    print(f"  Canal lateral temporal : "
          f"{end_to_end['timing_side_channel_ms']:.1f} ms")

    print(f"\n  [3/6] Parametros Argon2...")
    argon2_results = benchmark_argon2_params()
    for r in argon2_results:
        default = " <- PDF default" if r["is_default"] else ""
        print(f"  m={r['m']:6d} t={r['t']} p={r['p']}: "
              f"{r['time_ms_mean']:.0f} ms{default}")

    print(f"\n  [4/6] Memoria RAM (enrolamiento y autenticacion)...")
    memory = measure_memory(enroll_by_class, test_by_class,
                            args.n_bits, args.n_parity, args.n_enroll,
                            args.n_verify, seed=args.seed)
    print(f"  Pico enrolamiento : {memory['enroll']['peak_kb']:.1f} KB")
    print(f"  Pico autenticacion: {memory['auth']['peak_kb']:.1f} KB")
    sample_record = memory.pop("_record", None)

    print(f"\n  [5/6] Almacenamiento del template (medido)...")
    storage = measure_template_storage(args.n_bits, args.n_parity,
                                        record=sample_record)
    print(f"  Estimado : {storage['total_bytes']} B ({storage['total_kb']} KB)")
    if storage.get("measured_sqlite_bytes_per_user"):
        print(f"  SQLite   : {storage['measured_sqlite_bytes_per_user']} B/usuario")
    if storage.get("total_bytes_optimised"):
        print(f"  Optimizado: {storage['total_bytes_optimised']} B "
              f"({storage['total_kb_optimised']} KB) con indices empaquetados")

    print(f"\n  [6/6] Latencia de identificacion 1:N...")
    ident_1n = benchmark_identification_1n(
        enroll_by_class, test_by_class, args.n_bits, args.n_parity,
        args.n_enroll, args.n_verify, db_sizes=tuple(args.db_sizes),
        seed=args.seed)
    for r in ident_1n:
        print(f"  N={r['db_size']:>5}: {r['worst_case_ms']:.1f} ms peor caso "
              f"({r['ms_per_record']:.3f} ms/registro)")

    results = {
        "run_name":       args.run_name,
        "date":           datetime.now().isoformat(),
        "seed":           args.seed,
        "hardware":       hw,
        "n_bits":         args.n_bits,
        "n_parity":       args.n_parity,
        "n_enroll":       args.n_enroll,
        "n_verify":       args.n_verify,
        "scope_note":     ("Los tiempos excluyen segmentacion, preprocesado y "
                           "extraccion de embeddings."),
        "components":     comp_times,
        "end_to_end":     end_to_end,
        "identification_1n": ident_1n,
        "argon2_params":  argon2_results,
        "storage":        storage,
        "memory":         memory,
    }
    json_path = _ec.save_json(
        results, Path(results_dir) / "eval_computational_results.json")
    print(f"\n  ✓ JSON guardado")

    print("\n  Generando figuras...")
    fig_break = plot_breakdown(comp_times, end_to_end, results_dir)
    fig_arg   = plot_argon2(argon2_results, results_dir)

    print("\n  Generando PDF...")
    generate_pdf(args.run_name, comp_times, end_to_end, argon2_results,
                 storage, memory, fig_break, fig_arg, results_dir,
                 hw=hw, seed=args.seed, ident_1n=ident_1n,
                 n_bits=args.n_bits, n_parity=args.n_parity,
                 n_enroll=args.n_enroll, n_verify=args.n_verify)

    print(f"\n{'='*60}\n  COMPLETADO\n{'='*60}\n")


if __name__ == "__main__":
    main()
