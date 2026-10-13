"""
04_polar_transform.py
Transformación polar del iris usando el modelo rubber sheet de Daugman
implementado en el repositorio CVRL de Notre Dame.

Para cada imagen:
1. Carga la imagen preprocesada y su máscara segmentada
2. Detecta los círculos de pupila e iris con el ResNet18 de CVRL
3. Aplica la transformación cartesiana → polar (rubber sheet)
4. Guarda imagen_polar (64×512) y mascara_polar (64×512) en data/polar/

La máscara polar indica qué zonas del iris son válidas (no ocluidas
por párpados o pestañas) para la extracción de características.
"""

import os
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
    Carga el modelo CVRL con soporte para circApprox y cartToPol_torch.
    """
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

    print("  ✓ Modelo CVRL cargado en CUDA")
    return model


# ─────────────────────────────────────────────
# 3. TRANSFORMACIÓN POLAR DE UNA IMAGEN
# ─────────────────────────────────────────────

def polar_transform(model, image_path, mask_path):
    """
    Aplica la transformación polar a una imagen de iris.
    """
    try:
        image = Image.open(str(image_path)).convert('L')

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None

        pupil_xyr, iris_xyr = model.circApprox(image)

        image_polar, mask_polar = model.cartToPol_torch(
            image, mask, pupil_xyr, iris_xyr
        )

        return image_polar, mask_polar

    except Exception as e:
        print(f"      Error: {e}")
        return None, None


# ─────────────────────────────────────────────
# 4. PROCESADO POR DATASET
# ─────────────────────────────────────────────

def transform_dataset(model, processed_root, segmented_root, polar_root, dataset_name, extension="png"):
    """
    Aplica la transformación polar a todas las imágenes de un dataset.
    Guarda imagen_polar y mascara_polar manteniendo la estructura de carpetas.
    """
    input_dir     = Path(processed_root) / dataset_name
    segmented_dir = Path(segmented_root) / dataset_name
    polar_dir     = Path(polar_root) / dataset_name

    all_images = list(input_dir.glob(f"**/*.{extension}"))
    errors  = []
    skipped = 0

    for img_path in tqdm(all_images, desc=f"{dataset_name.upper():<8}"):
        relative  = img_path.relative_to(input_dir)

        mask_path = segmented_dir / relative.parent / (img_path.stem + "_mask.png")
        if not mask_path.exists():
            errors.append(f"Sin máscara: {img_path}")
            continue

        polar_img_path  = polar_dir / relative.parent / (img_path.stem + "_polar.png")
        polar_mask_path = polar_dir / relative.parent / (img_path.stem + "_polar_mask.png")
        polar_img_path.parent.mkdir(parents=True, exist_ok=True)

        if polar_img_path.exists() and polar_mask_path.exists():
            skipped += 1
            continue

        image_polar, mask_polar = polar_transform(model, img_path, mask_path)

        if image_polar is None:
            errors.append(str(img_path))
            continue

        cv2.imwrite(str(polar_img_path),  image_polar)
        cv2.imwrite(str(polar_mask_path), mask_polar)

    if skipped > 0:
        print(f"  → {skipped} imágenes ya procesadas, saltadas")
    if errors:
        print(f"  → {len(errors)} errores")

    return errors


# ─────────────────────────────────────────────
# 5. VERIFICACIÓN VISUAL
# ─────────────────────────────────────────────

def verify_polar(polar_root, results_dir):
    """
    Guarda una figura con imagen_polar y mascara_polar para
    una muestra de cada dataset.
    """
    import matplotlib.pyplot as plt

    datasets = ["casia", "iitd", "polyu", "ubiris"]
    fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 12))
    fig.suptitle("Verificación transformación polar — CVRL", fontsize=13)

    for i, dataset in enumerate(datasets):
        polars = list((Path(polar_root) / dataset).glob("**/*_polar.png"))
        polars = [p for p in polars if "_polar_mask" not in p.name]
        if not polars:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            axes[i, 0].set_title(f"{dataset.upper()} - (sin datos)")
            continue

        polar_path = polars[0]
        mask_path  = Path(str(polar_path).replace("_polar.png", "_polar_mask.png"))

        img  = cv2.imread(str(polar_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img,  cmap="gray", aspect="auto")
        axes[i, 0].set_title(f"{dataset.upper()} - Iris polar")
        axes[i, 0].axis("off")

        if mask is not None:
            axes[i, 1].imshow(mask, cmap="gray", aspect="auto")
        axes[i, 1].set_title(f"{dataset.upper()} - Máscara polar")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = Path(results_dir) / "polar_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Verificación guardada en {out_path}")


# ─────────────────────────────────────────────
# 6. MODO TEST
# ─────────────────────────────────────────────

def run_test(config):
    """
    Procesa 4 imágenes aleatorias de cada dataset y muestra el resultado.
    """
    import matplotlib.pyplot as plt

    processed_root  = config["paths"]["data"]["processed"]
    segmented_root  = config["paths"]["data"]["segmented"]
    results_dir     = config["paths"]["results"]
    test_output_dir = Path(results_dir) / "test_polar"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 Cargando modelo CVRL...")
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

        samples = random.sample(all_images, min(4, len(all_images)))
        print(f"\n🧪 {dataset_name} — {len(samples)} imágenes...")

        fig, axes = plt.subplots(len(samples), 3, figsize=(14, len(samples) * 3))
        fig.suptitle(f"Test transformación polar — {dataset_name}", fontsize=13)

        for i, img_path in enumerate(samples):
            rel            = img_path.relative_to(Path(processed_root))
            dataset_folder = rel.parts[0]
            rest           = Path(*rel.parts[1:])
            mask_path      = Path(segmented_root) / dataset_folder / rest.parent / (img_path.stem + "_mask.png")

            if not mask_path.exists():
                print(f"  ⚠ Sin máscara: {img_path.name}")
                continue

            image_polar, mask_polar = polar_transform(model, img_path, mask_path)

            if image_polar is None:
                print(f"  ⚠ Error: {img_path.name}")
                continue

            valid_pct = np.sum(mask_polar > 0) / mask_polar.size * 100
            print(f"  [{i+1}/{len(samples)}] {img_path.name}: polar={image_polar.shape}, válido={valid_pct:.1f}%")

            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            axes[i, 0].imshow(img_gray, cmap="gray")
            axes[i, 0].set_title(f"Original: {img_path.name}", fontsize=7)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(image_polar, cmap="gray", aspect="auto")
            axes[i, 1].set_title(f"Iris polar (64×512)", fontsize=7)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(mask_polar, cmap="gray", aspect="auto")
            axes[i, 2].set_title(f"Máscara polar — válido={valid_pct:.1f}%", fontsize=7)
            axes[i, 2].axis("off")

        plt.tight_layout()
        out_path = test_output_dir / f"test_polar_{dataset_name.lower()}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardado en {out_path}")

    print("\n✅ Test completado.\n")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser("Transformación polar del iris con CVRL")
    parser.add_argument("--test", action="store_true",
                        help="Modo test: 4 imágenes aleatorias de cada dataset")
    args = parser.parse_args()

    config = load_config("config.yaml")

    if args.test:
        run_test(config)
        return

    processed_root = config["paths"]["data"]["processed"]
    segmented_root = config["paths"]["data"]["segmented"]
    polar_root     = config["paths"]["data"]["polar"]
    results_dir    = config["paths"]["results"]

    Path(polar_root).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("\n📦 Cargando modelo CVRL...")
    model = load_cvrl_model()

    print("\n🔄 Iniciando transformación polar...")
    all_errors = []
    all_errors += transform_dataset(model, processed_root, segmented_root, polar_root, "casia")
    all_errors += transform_dataset(model, processed_root, segmented_root, polar_root, "iitd")
    all_errors += transform_dataset(model, processed_root, segmented_root, polar_root, "polyu")
    all_errors += transform_dataset(model, processed_root, segmented_root, polar_root, "ubiris")

    print("\n🖼️  Generando verificación visual...")
    verify_polar(polar_root, results_dir)

    if all_errors:
        print(f"\n  ⚠ Errores: {len(all_errors)}")
        log_path = Path(results_dir) / "polar_errors.txt"
        with open(log_path, "w") as f:
            for e in all_errors:
                f.write(e + "\n")
        print(f"  ✓ Log guardado en {log_path}")
    else:
        print(f"\n  ✓ Sin errores")

    print("\n✅ Transformación polar completada.\n")


if __name__ == "__main__":
    main()
