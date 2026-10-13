"""
03_segment.py
Segmentación automática del iris usando el modelo U-Net++ de Notre Dame (CVRL).
- Modelo NestedSharedAtrousResUNet entrenado con CASIA-V4, IITD y otros datasets
- No necesita bounding box como prompt
- Funciona con CASIA, IITD, PolyU (NIR y VIS) y UBIRIS (VIS)
- ~0.025s por imagen en GPU
"""

import sys
import cv2
import yaml
import math
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Añadir el repositorio CVRL al path
CVRL_REPO = Path(__file__).parent.parent / "models/cvrl_iris/methods/HDBIF/Python"


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. CARGA DEL MODELO
# ─────────────────────────────────────────────

def load_cvrl_model():
    """
    Carga el modelo de segmentación U-Net++ de Notre Dame (CVRL).
    """
    import os

    original_dir = os.getcwd()
    os.chdir(str(CVRL_REPO))

    sys.path.insert(0, str(CVRL_REPO))
    sys.path.insert(0, str(CVRL_REPO / "modules"))

    from irisRecognition import irisRecognition

    cfg_path = CVRL_REPO / "cfg.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["mask_model_path"]   = str(CVRL_REPO / cfg["mask_model_path"].lstrip("./"))
    cfg["circle_model_path"] = str(CVRL_REPO / cfg["circle_model_path"].lstrip("./"))
    cfg["recog_bsif_dir"]    = str(CVRL_REPO / cfg["recog_bsif_dir"].lstrip("./")) + "/"
    cfg["cuda"] = True

    model = irisRecognition(cfg)

    os.chdir(original_dir)

    print("  ✓ Modelo CVRL U-Net++ cargado en CUDA")
    return model


# ─────────────────────────────────────────────
# 3. SEGMENTACIÓN DE UNA IMAGEN
# ─────────────────────────────────────────────

def segment_image(model, image_path):
    try:
        image = Image.open(str(image_path)).convert('L')
        mask = model.segment(image)
        return mask
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# 4. PROCESADO POR DATASET
# ─────────────────────────────────────────────

def segment_dataset(model, processed_root, segmented_root, dataset_name, extension="png"):
    """
    Segmenta todas las imágenes de un dataset.
    Mantiene la misma estructura de carpetas.
    """
    input_dir  = Path(processed_root) / dataset_name
    output_dir = Path(segmented_root) / dataset_name

    all_images = list(input_dir.glob(f"**/*.{extension}"))
    errors  = []
    skipped = 0

    for img_path in tqdm(all_images, desc=f"{dataset_name.upper():<8}"):
        relative  = img_path.relative_to(input_dir)
        mask_path = output_dir / relative.parent / (img_path.stem + "_mask.png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if mask_path.exists():
            skipped += 1
            continue

        mask = segment_image(model, img_path)

        if mask is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(mask_path), mask)

    if skipped > 0:
        print(f"  → {skipped} imágenes ya procesadas, saltadas")
    if errors:
        print(f"  → {len(errors)} errores")

    return errors


# ─────────────────────────────────────────────
# 5. VERIFICACIÓN VISUAL
# ─────────────────────────────────────────────

def verify_segmentation(processed_root, segmented_root, results_dir):
    """
    Guarda una figura con original + máscara para una muestra de cada dataset.
    """
    import matplotlib.pyplot as plt

    datasets = ["casia", "iitd", "polyu", "ubiris"]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 16))
    fig.suptitle("Verificación de segmentación — CVRL U-Net++", fontsize=13)

    for i, dataset in enumerate(datasets):
        masks = list((Path(segmented_root) / dataset).glob("**/*_mask.png"))
        if not masks:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 0].set_title(f"{dataset.upper()} - (sin datos)")
            continue

        mask_path = masks[0]
        relative  = mask_path.relative_to(Path(segmented_root) / dataset)
        orig_name = mask_path.stem.replace("_mask", "") + ".png"
        orig_path = Path(processed_root) / dataset / relative.parent / orig_name

        img  = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img,  cmap="gray")
        axes[i, 0].set_title(f"{dataset.upper()} - Original")
        axes[i, 0].axis("off")

        if img is not None and mask is not None:
            overlay = img.copy()
            overlay[mask > 127] = 255
            axes[i, 1].imshow(overlay, cmap="gray")
        axes[i, 1].set_title(f"{dataset.upper()} - Máscara")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = Path(results_dir) / "segmentation_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Verificación guardada en {out_path}")


# ─────────────────────────────────────────────
# 6. MODO TEST
# ─────────────────────────────────────────────

def run_test(config):
    """
    Procesa 10 imágenes aleatorias de cada dataset.
    Muestra los resultados en cuadrícula.
    """
    import matplotlib.pyplot as plt

    processed_root  = config["paths"]["data"]["processed"]
    results_dir     = config["paths"]["results"]
    test_output_dir = Path(results_dir) / "test_segmentation"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 Cargando modelo CVRL U-Net++...")
    model = load_cvrl_model()

    datasets = {
        "CASIA":     list(Path(processed_root).glob("casia/*/L/*.png")),
        "IITD":      list(Path(processed_root).glob("iitd/*/*.png")),
        "PolyU_NIR": list(Path(processed_root).glob("polyu/*/L/NIR/*.png")),
        "PolyU_VIS": list(Path(processed_root).glob("polyu/*/L/VIS/*.png")),
        "UBIRIS":    list(Path(processed_root).glob("ubiris/*/*.png")),
    }

    for dataset_name, all_images in datasets.items():
        if not all_images:
            print(f"  ⚠ No hay imágenes para {dataset_name}")
            continue

        samples = random.sample(all_images, min(10, len(all_images)))
        print(f"\n🧪 {dataset_name} — {len(samples)} imágenes...")

        n_rows = math.ceil(len(samples) / 2)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3.5))
        fig.suptitle(f"Test segmentación CVRL — {dataset_name}", fontsize=14)

        for i, img_path in enumerate(samples):
            row = i // 2
            col = (i % 2) * 2

            mask = segment_image(model, img_path)
            if mask is None:
                print(f"  ⚠ Error: {img_path.name}")
                continue

            area = np.sum(mask > 0) / (640 * 480) * 100
            print(f"  [{i+1}/{len(samples)}] {img_path.name}: área={area:.1f}%")

            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(img_gray, cmap="gray")
            axes[row, col].set_title(img_path.name, fontsize=7)
            axes[row, col].axis("off")

            axes[row, col+1].imshow(mask, cmap="gray")
            axes[row, col+1].set_title(f"área={area:.1f}%", fontsize=7)
            axes[row, col+1].axis("off")

        plt.tight_layout()
        out_path = test_output_dir / f"test_{dataset_name.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardado en {out_path}")

    print("\n✅ Test completado.\n")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser("Segmentación de iris con CVRL U-Net++")
    parser.add_argument("--test", action="store_true",
                        help="Modo test: 10 imágenes aleatorias de cada dataset")
    args = parser.parse_args()

    config = load_config("config.yaml")

    if args.test:
        run_test(config)
        return

    processed_root = config["paths"]["data"]["processed"]
    segmented_root = config["paths"]["data"]["segmented"]
    results_dir    = config["paths"]["results"]

    Path(segmented_root).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("\n📦 Cargando modelo CVRL U-Net++...")
    model = load_cvrl_model()

    print("\n🔍 Iniciando segmentación...")
    all_errors = []
    all_errors += segment_dataset(model, processed_root, segmented_root, "casia")
    all_errors += segment_dataset(model, processed_root, segmented_root, "iitd")
    all_errors += segment_dataset(model, processed_root, segmented_root, "polyu")
    all_errors += segment_dataset(model, processed_root, segmented_root, "ubiris")

    print("\n🖼️  Generando verificación visual...")
    verify_segmentation(processed_root, segmented_root, results_dir)

    if all_errors:
        print(f"\n  ⚠ Errores: {len(all_errors)}")
        log_path = Path(results_dir) / "segmentation_errors.txt"
        with open(log_path, "w") as f:
            for e in all_errors:
                f.write(e + "\n")
        print(f"  ✓ Log guardado en {log_path}")
    else:
        print(f"\n  ✓ Sin errores")

    print("\n✅ Segmentación completada.\n")


if __name__ == "__main__":
    main()
