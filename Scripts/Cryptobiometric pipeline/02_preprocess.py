"""
02_preprocess.py
Normalización ISO de las imágenes de los cuatro datasets.
- Convierte todas las imágenes a escala de grises
- Redimensiona a 640x480 (ratio 4:3)
- Normaliza la iluminación con CLAHE
- Guarda en formato PNG con estructura homogénea
"""

import os
import re
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. PROCESADO DE UNA IMAGEN
# ─────────────────────────────────────────────

def preprocess_image(img_path, target_size=(640, 480)):
    """
    Carga, convierte a gris, redimensiona y normaliza una imagen.
    Devuelve la imagen procesada o None si hay error.
    """
    # Cargar imagen (UNCHANGED para tiff de 16 bits)
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Convertir a 8 bits si es de 16 bits (algunos tiff)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)

    # Convertir a escala de grises si es color (PolyU VIS, UBIRIS VIS)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 640x480
    if img.shape != (target_size[1], target_size[0]):
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Normalización de iluminación con CLAHE
    # CLAHE (Contrast Limited Adaptive Histogram Equalization):
    # Mejora el contraste localmente sin saturar zonas brillantes.
    # Parámetros: clipLimit=2.0 (límite de contraste), tileGridSize=8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


# ─────────────────────────────────────────────
# 3. PROCESADO POR DATASET
# ─────────────────────────────────────────────

def process_casia(raw_root, out_root, target_size):
    """
    Estructura entrada:  <raw_root>/<sujeto>/<L|R>/<imagen>.jpg
    Estructura salida:   <out_root>/casia/<sujeto>/<L|R>/<imagen>.png
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root) / "casia"
    errors = []

    all_images = list(raw_root.glob("*/*/*.jpg"))

    for img_path in tqdm(all_images, desc="CASIA "):
        relative = img_path.relative_to(raw_root)  # ej: 000/L/S5000L00.jpg
        out_path = out_root / relative.parent / (img_path.stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        img = preprocess_image(img_path, target_size)
        if img is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(out_path), img)

    return errors


def process_iitd(raw_root, out_root, target_size):
    """
    Estructura entrada:  <raw_root>/<sujeto>/<muestra>_<L|R>.bmp
    Estructura salida:   <out_root>/iitd/<sujeto>/<muestra>_<L|R>.png
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root) / "iitd"
    errors = []

    all_images = list(raw_root.glob("*/*.bmp"))

    for img_path in tqdm(all_images, desc="IITD  "):
        relative = img_path.relative_to(raw_root)  # ej: 001/01_L.bmp
        out_path = out_root / relative.parent / (img_path.stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        img = preprocess_image(img_path, target_size)
        if img is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(out_path), img)

    return errors


def process_polyu(raw_root, out_root, target_size):
    """
    Estructura entrada:  <raw_root>/<sujeto>/<L|R>/<NIR|VIS>/<imagen>.tiff
    Estructura salida:   <out_root>/polyu/<sujeto>/<L|R>/<NIR|VIS>/<imagen>.png
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root) / "polyu"
    errors = []

    all_images = list(raw_root.glob("*/*/*/*tiff"))

    for img_path in tqdm(all_images, desc="PolyU "):
        relative = img_path.relative_to(raw_root)  # ej: 001/L/NIR/001_L_NIR_1.tiff
        out_path = out_root / relative.parent / (img_path.stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        img = preprocess_image(img_path, target_size)
        if img is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(out_path), img)

    return errors


def process_ubiris(raw_root, out_root, target_size):
    """
    Estructura entrada:  <raw_root>/C{eye_id}_S{session}_I{image}.tiff  (carpeta plana)
    Estructura salida:   <out_root>/ubiris/<eye_id>/C{eye_id}_S{session}_I{image}.png

    Convención eye_id:
      - par   → ojo izquierdo (L)
      - impar → ojo derecho (R)
      - eye_id 2k y 2k+1 pertenecen al mismo sujeto k
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root) / "ubiris"
    errors = []
    skipped = 0

    all_images = list(raw_root.glob("*.tiff"))

    for img_path in tqdm(all_images, desc="UBIRIS"):
        match = re.match(r"C(\d+)_S(\d+)_I(\d+)\.tiff", img_path.name, re.IGNORECASE)
        if not match:
            skipped += 1
            continue

        eye_id = match.group(1)  # conservar como string para mantener ceros

        # Salida: ubiris/<eye_id>/<nombre>.png
        out_path = out_root / eye_id / (img_path.stem + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        img = preprocess_image(img_path, target_size)
        if img is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(out_path), img)

    if skipped:
        print(f"  ⚠ UBIRIS: {skipped} archivo(s) ignorados por nombre no reconocido")

    return errors


# ─────────────────────────────────────────────
# 4. VERIFICACIÓN POST-PROCESADO
# ─────────────────────────────────────────────

def verify_output(processed_root):
    """
    Cuenta imágenes procesadas y verifica que tienen el tamaño correcto.
    """
    processed_root = Path(processed_root)
    stats = {}

    for dataset in ["casia", "iitd", "polyu", "ubiris"]:
        dataset_dir = processed_root / dataset
        if not dataset_dir.exists():
            continue

        images = list(dataset_dir.glob("**/*.png"))
        wrong_size = []

        for img_path in images[:20]:  # verificar muestra
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.shape != (480, 640):
                wrong_size.append(str(img_path))

        stats[dataset] = {
            "total": len(images),
            "wrong_size": len(wrong_size)
        }

    return stats


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    config = load_config("config.yaml")

    raw_root       = config["paths"]["data"]["raw"]
    processed_root = config["paths"]["data"]["processed"]
    target_size    = tuple(config["preprocessing"]["image_size"])  # (640, 480)

    Path(processed_root).mkdir(parents=True, exist_ok=True)

    print("\n🔄 Iniciando preprocesado de imágenes...")
    print(f"   Tamaño objetivo : {target_size[0]}×{target_size[1]}")
    print(f"   Salida          : {processed_root}\n")

    all_errors = []

    # Procesar cada dataset
    all_errors += process_casia (f"{raw_root}/casia",  processed_root, target_size)
    all_errors += process_iitd  (f"{raw_root}/iitd",   processed_root, target_size)
    all_errors += process_polyu (f"{raw_root}/polyu",  processed_root, target_size)
    all_errors += process_ubiris(f"{raw_root}/ubiris", processed_root, target_size)

    # Verificar resultado
    print("\n🔍 Verificando salida...")
    stats = verify_output(processed_root)
    for dataset, info in stats.items():
        status = "✓" if info["wrong_size"] == 0 else "⚠"
        print(f"  {status} {dataset.upper():<8} → {info['total']:>6} imágenes procesadas", end="")
        if info["wrong_size"] > 0:
            print(f"  ({info['wrong_size']} con tamaño incorrecto)", end="")
        print()

    # Reportar errores
    if all_errors:
        print(f"\n  ⚠ Imágenes con error: {len(all_errors)}")
        for e in all_errors:
            print(f"    - {e}")
    else:
        print(f"\n  ✓ Sin errores")

    print("\n✅ Preprocesado completado.\n")


if __name__ == "__main__":
    main()
