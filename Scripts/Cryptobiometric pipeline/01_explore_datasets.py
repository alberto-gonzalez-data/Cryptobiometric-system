"""
01_explore_datasets.py
Exploración y análisis de los datasets de iris: CASIA, IITD, PolyU y UBIRIS.
Genera estadísticas, detecta anomalías y guarda un resumen en results/.
"""

import os
import re
import csv
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. PARSEO DE DATASETS
# ─────────────────────────────────────────────

def parse_casia(root):
    """
    Estructura: <root>/<sujeto>/<L|R>/S<sujeto><ojo><muestra>.jpg
    Devuelve lista de dicts con info de cada imagen.
    """
    records = []
    root = Path(root)

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name  # "000" .. "999"

        for eye_dir in sorted(subject_dir.iterdir()):
            if not eye_dir.is_dir():
                continue
            eye = eye_dir.name  # "L" o "R"

            for img_path in sorted(eye_dir.glob("*.jpg")):
                # Patrón: S5000L00.jpg
                match = re.match(r"S(\d+)([LR])(\d+)\.jpg", img_path.name)
                sample = match.group(3) if match else "?"
                records.append({
                    "dataset": "CASIA",
                    "subject": subject_id,
                    "eye": eye,
                    "spectrum": "NIR",
                    "sample": sample,
                    "path": str(img_path),
                    "format": "jpg"
                })

    return records


def parse_iitd(root):
    """
    Estructura: <root>/<sujeto>/<muestra>_<L|R>.bmp
    Devuelve lista de dicts con info de cada imagen.
    """
    records = []
    root = Path(root)

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name  # "001" .. "224"

        for img_path in sorted(subject_dir.glob("*.bmp")):
            # Patrón: 01_L.bmp
            match = re.match(r"(\d+)_([LR])\.bmp", img_path.name)
            if match:
                sample = match.group(1)
                eye = match.group(2)
            else:
                sample, eye = "?", "?"

            records.append({
                "dataset": "IITD",
                "subject": subject_id,
                "eye": eye,
                "spectrum": "NIR",
                "sample": sample,
                "path": str(img_path),
                "format": "bmp"
            })

    return records


def parse_polyu(root):
    """
    Estructura: <root>/<sujeto>/<L|R>/<NIR|VIS>/<sujeto>_<ojo>_<espectro>_<muestra>.tiff
    Devuelve lista de dicts con info de cada imagen.
    """
    records = []
    root = Path(root)

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name  # "001" .. "106"

        for eye_dir in sorted(subject_dir.iterdir()):
            if not eye_dir.is_dir():
                continue
            eye = eye_dir.name  # "L" o "R"

            for spectrum_dir in sorted(eye_dir.iterdir()):
                if not spectrum_dir.is_dir():
                    continue
                spectrum = spectrum_dir.name  # "NIR" o "VIS"

                for img_path in sorted(spectrum_dir.glob("*.tiff")):
                    # Patrón: 001_L_NIR_1.tiff
                    match = re.match(r"(\d+)_([LR])_(NIR|VIS)_(\d+)\.tiff", img_path.name)
                    sample = match.group(4) if match else "?"
                    records.append({
                        "dataset": "PolyU",
                        "subject": subject_id,
                        "eye": eye,
                        "spectrum": spectrum,
                        "sample": sample,
                        "path": str(img_path),
                        "format": "tiff"
                    })

    return records


def parse_ubiris(root):
    """
    Estructura: <root>/ (carpeta plana con todos los .tiff)
    Naming: C{eye_id}_S{session}_I{image}.tiff

    Convención:
      - eye_id par  → ojo izquierdo (L)
      - eye_id impar → ojo derecho (R)
      - eye_id = 2k y 2k+1 pertenecen al mismo sujeto k

    Devuelve lista de dicts con info de cada imagen.
    """
    records = []
    root = Path(root)

    for img_path in sorted(root.glob("*.tiff")):
        # Patrón: C002_S1_I01.tiff
        match = re.match(r"C(\d+)_S(\d+)_I(\d+)\.tiff", img_path.name, re.IGNORECASE)
        if not match:
            print(f"  ⚠ Nombre no reconocido: {img_path.name}")
            continue

        eye_id  = int(match.group(1))
        session = match.group(2)
        sample  = match.group(3)

        # Derivar sujeto y ojo a partir de eye_id
        subject_id = str(eye_id // 2)          # sujeto k
        eye        = "L" if eye_id % 2 == 0 else "R"

        records.append({
            "dataset":  "UBIRIS",
            "subject":  subject_id,
            "eye_id":   str(eye_id),            # id único del ojo (útil para clases)
            "eye":      eye,
            "spectrum": "VIS",
            "session":  session,
            "sample":   sample,
            "path":     str(img_path),
            "format":   "tiff"
        })

    return records


# ─────────────────────────────────────────────
# 3. ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────

def analyze(records, dataset_name):
    """Calcula estadísticas generales de un dataset."""
    subjects = defaultdict(list)
    for r in records:
        subjects[r["subject"]].append(r)

    n_subjects = len(subjects)
    n_images = len(records)
    samples_per_subject = [len(v) for v in subjects.values()]

    eyes = defaultdict(int)
    spectra = defaultdict(int)
    for r in records:
        eyes[r["eye"]] += 1
        spectra[r["spectrum"]] += 1

    # Sujetos con menos de 4 imágenes (posible anomalía)
    low_sample_subjects = [k for k, v in subjects.items() if len(v) < 4]

    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*50}")
    print(f"  Sujetos totales       : {n_subjects}")
    print(f"  Imágenes totales      : {n_images}")
    print(f"  Imágenes/sujeto (min) : {min(samples_per_subject)}")
    print(f"  Imágenes/sujeto (max) : {max(samples_per_subject)}")
    print(f"  Imágenes/sujeto (med) : {np.mean(samples_per_subject):.1f}")
    print(f"  Distribución ojos     : {dict(eyes)}")
    print(f"  Espectros             : {dict(spectra)}")

    # UBIRIS: mostrar también distribución por sesión
    if dataset_name == "UBIRIS":
        sessions = defaultdict(int)
        eye_ids  = set()
        for r in records:
            sessions[r["session"]] += 1
            eye_ids.add(r["eye_id"])
        print(f"  Ojos únicos (eye_id)  : {len(eye_ids)}")
        print(f"  Distribución sesiones : { {f'S{k}': v for k, v in sorted(sessions.items())} }")

    if low_sample_subjects:
        print(f"  ⚠ Sujetos con <4 imgs : {len(low_sample_subjects)} → {low_sample_subjects[:5]}...")
    else:
        print(f"  ✓ Sin anomalías de muestras bajas")

    return {
        "dataset": dataset_name,
        "n_subjects": n_subjects,
        "n_images": n_images,
        "min_samples": min(samples_per_subject),
        "max_samples": max(samples_per_subject),
        "mean_samples": round(np.mean(samples_per_subject), 1),
        "eyes": dict(eyes),
        "spectra": dict(spectra),
        "low_sample_subjects": len(low_sample_subjects)
    }


# ─────────────────────────────────────────────
# 4. VERIFICACIÓN DE IMÁGENES
# ─────────────────────────────────────────────

def check_images(records, dataset_name, sample_size=50):
    """
    Verifica que las imágenes se pueden leer correctamente.
    Comprueba una muestra aleatoria para no tardar demasiado.
    """
    print(f"\n  Verificando imágenes ({sample_size} muestras aleatorias)...")
    indices = np.random.choice(len(records), min(sample_size, len(records)), replace=False)
    corrupt = []

    for i in indices:
        path = records[i]["path"]
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                corrupt.append(path)
        except Exception as e:
            corrupt.append(path)

    if corrupt:
        print(f"  ⚠ Imágenes corruptas encontradas: {len(corrupt)}")
        for p in corrupt:
            print(f"    - {p}")
    else:
        print(f"  ✓ Todas las imágenes verificadas correctamente")

    return corrupt


# ─────────────────────────────────────────────
# 5. VISUALIZACIÓN DE EJEMPLOS
# ─────────────────────────────────────────────

def visualize_samples(all_records, results_dir):
    """
    Muestra una imagen de ejemplo de cada dataset/espectro
    y guarda la figura en results/.
    """
    samples_to_show = [
        ("CASIA",  "NIR"),
        ("IITD",   "NIR"),
        ("PolyU",  "NIR"),
        ("PolyU",  "VIS"),
        ("UBIRIS", "VIS"),
    ]

    fig, axes = plt.subplots(1, len(samples_to_show), figsize=(20, 4))
    fig.suptitle("Ejemplos de imágenes por dataset y espectro", fontsize=13)

    for ax, (dataset, spectrum) in zip(axes, samples_to_show):
        candidates = [r for r in all_records
                      if r["dataset"] == dataset and r["spectrum"] == spectrum]
        if not candidates:
            ax.axis("off")
            ax.set_title(f"{dataset}\n{spectrum}\n(sin datos)")
            continue

        sample = candidates[0]
        img = cv2.imread(sample["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(sample["path"], cv2.IMREAD_UNCHANGED)

        if img is not None:
            ax.imshow(img, cmap="gray")
            label = f"{dataset} · {spectrum}\nSujeto {sample['subject']} · {sample['eye']}"
            if dataset == "UBIRIS":
                label += f"\neye_id={sample['eye_id']} · S{sample['session']}"
            ax.set_title(label)
        else:
            ax.set_title(f"{dataset} · {spectrum}\n(error al cargar)")
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(results_dir) / "dataset_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  ✓ Figura guardada en {out_path}")


# ─────────────────────────────────────────────
# 6. GUARDAR ESTADÍSTICAS EN CSV
# ─────────────────────────────────────────────

def save_stats(stats_list, results_dir):
    out_path = Path(results_dir) / "dataset_stats.csv"
    keys = ["dataset", "n_subjects", "n_images", "min_samples",
            "max_samples", "mean_samples", "low_sample_subjects"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stats_list)

    print(f"\n  ✓ Estadísticas guardadas en {out_path}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    config = load_config("config.yaml")

    raw_root    = config["paths"]["data"]["raw"]
    casia_root  = raw_root + "/casia"
    iitd_root   = raw_root + "/iitd"
    polyu_root  = raw_root + "/polyu"
    ubiris_root = raw_root + "/ubiris"
    results_dir = config["paths"]["results"]

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("\n📂 Parseando datasets...")
    casia_records  = parse_casia(casia_root)
    iitd_records   = parse_iitd(iitd_root)
    polyu_records  = parse_polyu(polyu_root)
    ubiris_records = parse_ubiris(ubiris_root)
    all_records    = casia_records + iitd_records + polyu_records + ubiris_records

    print(f"\n  CASIA  → {len(casia_records)} imágenes encontradas")
    print(f"  IITD   → {len(iitd_records)} imágenes encontradas")
    print(f"  PolyU  → {len(polyu_records)} imágenes encontradas")
    print(f"  UBIRIS → {len(ubiris_records)} imágenes encontradas")
    print(f"  TOTAL  → {len(all_records)} imágenes")

    print("\n📊 Analizando estadísticas...")
    stats = []
    stats.append(analyze(casia_records,  "CASIA"))
    stats.append(analyze(iitd_records,   "IITD"))
    stats.append(analyze(polyu_records,  "PolyU"))
    stats.append(analyze(ubiris_records, "UBIRIS"))

    print("\n🔍 Verificando integridad de imágenes...")
    check_images(casia_records,  "CASIA")
    check_images(iitd_records,   "IITD")
    check_images(polyu_records,  "PolyU")
    check_images(ubiris_records, "UBIRIS")

    print("\n🖼️  Generando visualización de ejemplos...")
    visualize_samples(all_records, results_dir)

    print("\n💾 Guardando estadísticas...")
    save_stats(stats, results_dir)

    print("\n✅ Exploración completada.\n")


if __name__ == "__main__":
    main()
