"""
06_cancelable.py
Transformación biométrica cancelable para el sistema criptobiométrico de iris.

Pipeline:
  1. Binarización del embedding (umbral en cero)
  2. Selección de bits fiables (mayor magnitud) sobre el embedding medio
  3. Permutación determinista controlada por una semilla individual por
     usuario (transform_seed, 256 bits aleatorios)

Propiedades:
  - Revocabilidad: otra semilla genera un template completamente distinto
  - Unlinkability: semillas distintas producen templates no correlacionados
  - Determinismo: misma entrada y misma semilla -> mismo template

Nota: el sistema (07, 09 y la demo) genera una semilla por usuario. El modo de
línea de comandos de este módulo, solo para un análisis rápido de la BER, usa
una única semilla de sistema (--token_path) para todos.

Uso:
    python scripts/06_cancelable.py --run_name exp08
"""

import json
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import yaml


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_or_create_system_token(token_path: str) -> str:
    """
    Carga el token del sistema desde fichero o lo genera si no existe.
    El token es un valor aleatorio de 256 bits (64 hex chars) generado
    con secrets.token_hex. Se genera una unica vez y se reutiliza siempre.
    Perder este fichero invalida todos los templates del sistema.
    """
    import secrets
    token_path = Path(token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    if token_path.exists():
        token = token_path.read_text().strip()
        print(f"  \u2713 Token del sistema cargado desde {token_path}")
    else:
        token = secrets.token_hex(32)  # 256 bits aleatorios
        token_path.write_text(token)
        print(f"  \u2713 Token del sistema generado y guardado en {token_path}")
        print(f"    \u26a0  Guarda este fichero de forma segura.")

    return token


# ─────────────────────────────────────────────
# 2. BINARIZACIÓN
# ─────────────────────────────────────────────

def binarize(embedding: np.ndarray) -> np.ndarray:
    """
    Convierte un embedding continuo en un vector binario.
    Umbral en cero: componente > 0 → 1, componente <= 0 → 0.

    Los embeddings están normalizados en L2 y aproximadamente
    centrados en cero, por lo que este umbral produce vectores
    con ~50% de unos, óptimo para operaciones criptográficas.

    Args:
        embedding: vector float normalizado L2, shape (d,)

    Returns:
        vector binario uint8, shape (d,)
    """
    return (embedding > 0).astype(np.uint8)


# ─────────────────────────────────────────────
# 3. TRANSFORMACIÓN CANCELABLE
# ─────────────────────────────────────────────

def get_transform_seed(user_id: str, transform_seed: str) -> int:
    """
    Deriva un seed entero determinista combinando user_id y transform_seed.

    Cada usuario tiene su propio transform_seed aleatorio generado en el
    enrolamiento. Combinarlo con el user_id mediante SHA-256 garantiza:
      - Cada usuario tiene una permutación única
      - Si el transform_seed se revoca, la nueva permutación es completamente
        distinta e incorrelacionada con la anterior (revocabilidad)
      - Sin conocer el transform_seed no se puede reproducir la permutación

    Args:
        user_id:          identificador único del usuario
        transform_seed:   seed secreto individual del usuario (hex 64 chars)

    Returns:
        seed entero de 32 bits para np.random.Generator
    """
    material = f"{user_id}||{transform_seed}".encode('utf-8')
    digest   = hashlib.sha256(material).digest()
    seed     = int.from_bytes(digest[:4], byteorder='big')
    return seed


def apply_cancelable_transform(binary_vector: np.ndarray,
                                user_id: str,
                                transform_seed: str) -> np.ndarray:
    """
    Aplica una permutación aleatoria determinista al vector binario.

    La permutación depende del user_id y el transform_seed individual,
    garantizando unlinkability entre usuarios y revocabilidad individual.

    Args:
        binary_vector:  vector binario uint8, shape (d,)
        user_id:        identificador único del usuario
        transform_seed: seed secreto individual del usuario

    Returns:
        template cancelable binario uint8, shape (d,)
    """
    seed = get_transform_seed(user_id, transform_seed)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(binary_vector))
    return binary_vector[perm]


def invert_cancelable_transform(cancelable_vector: np.ndarray,
                                 user_id: str,
                                 transform_seed: str) -> np.ndarray:
    """
    Invierte la permutación cancelable (útil para análisis, no para producción).

    Args:
        cancelable_vector: template cancelable, shape (d,)
        user_id:           identificador único del usuario
        transform_seed:    seed secreto individual del usuario

    Returns:
        vector binario original antes de la permutación, shape (d,)
    """
    seed = get_transform_seed(user_id, transform_seed)
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(cancelable_vector))

    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return cancelable_vector[inv_perm]


# ─────────────────────────────────────────────
# 4. PIPELINE DE ENROLAMIENTO Y VERIFICACIÓN
# ─────────────────────────────────────────────

def enroll_template(embeddings: list,
                    user_id: str,
                    transform_seed: str) -> np.ndarray:
    """
    Genera el template cancelable de enrolamiento a partir de
    múltiples embeddings del mismo usuario.

    Pipeline:
      1. Media de todos los embeddings de enrolamiento
      2. Normalización L2
      3. Binarización
      4. Transformación cancelable con seed individual

    Args:
        embeddings:     lista de embeddings float normalizados L2
        user_id:        identificador único del usuario
        transform_seed: seed secreto individual del usuario

    Returns:
        template cancelable binario uint8, shape (d,)
    """
    mean_emb = np.mean(np.array(embeddings), axis=0)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    binary     = binarize(mean_emb)
    cancelable = apply_cancelable_transform(binary, user_id, transform_seed)
    return cancelable


def verify_template(embedding: np.ndarray,
                    user_id: str,
                    transform_seed: str) -> np.ndarray:
    """
    Genera el template cancelable de verificación a partir de
    un único embedding de consulta.

    Pipeline:
      1. Binarización del embedding
      2. Transformación cancelable con el mismo seed del enrolamiento

    Args:
        embedding:      embedding float normalizado L2, shape (d,)
        user_id:        identificador único del usuario
        transform_seed: seed secreto individual del usuario

    Returns:
        template cancelable binario uint8, shape (d,)
    """
    binary     = binarize(embedding)
    cancelable = apply_cancelable_transform(binary, user_id, transform_seed)
    return cancelable


def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia de Hamming normalizada entre dos vectores binarios.
    Devuelve la fracción de bits distintos en [0, 1].
    """
    return float(np.sum(a != b)) / len(a)


def select_reliable_bits(mean_embedding: np.ndarray,
                          n_bits: int) -> np.ndarray:
    """
    Selecciona los índices de los n_bits más estables del embedding.

    Los bits más estables son los que corresponden a componentes del
    embedding con mayor valor absoluto en la media de enrolamiento,
    es decir, los más alejados del umbral de binarización (cero).
    Cuanto más alejado del umbral, mayor variación necesita una captura
    para flipear ese bit, y por tanto más estable es entre capturas.

    Args:
        mean_embedding: media normalizada de los embeddings de enrolamiento
        n_bits:         número de bits a seleccionar

    Returns:
        índices ordenados de los n_bits más estables, shape (n_bits,)
    """
    # Ordenar por valor absoluto descendente y tomar los n_bits primeros
    stability = np.abs(mean_embedding)
    indices   = np.argsort(stability)[::-1][:n_bits]
    # Devolver ordenados para reproducibilidad
    return np.sort(indices)


def enroll_template_reliable(embeddings: list,
                              user_id: str,
                              transform_seed: str,
                              n_bits: int) -> tuple:
    """
    Genera el template cancelable de enrolamiento usando solo los bits fiables.

    Pipeline:
      1. Media normalizada de todos los embeddings de enrolamiento
      2. Selección de los n_bits más estables (mayor |valor| en la media)
      3. Binarización del subvector seleccionado
      4. Transformación cancelable con seed individual

    Args:
        embeddings:     lista de embeddings float normalizados L2
        user_id:        identificador único del usuario
        transform_seed: seed secreto individual del usuario
        n_bits:         número de bits fiables a usar

    Returns:
        (template cancelable binario, índices de bits fiables)
    """
    mean_emb = np.mean(np.array(embeddings), axis=0)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    indices    = select_reliable_bits(mean_emb, n_bits)
    subvector  = mean_emb[indices]
    binary     = binarize(subvector)
    cancelable = apply_cancelable_transform(binary, user_id, transform_seed)

    return cancelable, indices


def verify_template_reliable(embedding: np.ndarray,
                              user_id: str,
                              transform_seed: str,
                              indices: np.ndarray) -> np.ndarray:
    """
    Genera el template cancelable de verificación usando los mismos
    índices de bits fiables que se usaron en el enrolamiento.

    Args:
        embedding:      embedding float normalizado L2 de la imagen de verificación
        user_id:        identificador único del usuario
        transform_seed: seed secreto individual del usuario
        indices:        índices de bits fiables del enrolamiento

    Returns:
        template cancelable binario, shape (n_bits,)
    """
    subvector  = embedding[indices]
    binary     = binarize(subvector)
    cancelable = apply_cancelable_transform(binary, user_id, transform_seed)

    return cancelable


# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────

def load_split(features_root: str, split_name: str):
    """Carga embeddings, class_strings y paths de un split."""
    split_dir     = Path(features_root) / split_name
    embeddings    = np.load(str(split_dir / "embeddings.npy"))
    class_strings = json.load(open(str(split_dir / "class_strings.json")))
    return embeddings, class_strings


def evaluate(run_name: str, token: str, results_dir: str, features_root: str):
    """
    Evalúa la BER con binarización simple y con selección de bits fiables
    para distintos tamaños (512, 256, 128 bits).

    Permite elegir el tamaño óptimo para Reed-Solomon en el bloque 08:
      - Menor n_bits → menor BER genuina → RS más pequeño → secreto más grande
      - Mayor n_bits → mayor información biométrica → mejor discriminación
    """
    print(f"\n📂 Cargando embeddings de {run_name}...")

    # Detectar splits automáticamente
    import os
    possible = ["enroll_joint", "enroll_phase2", "enroll_phase1", "enroll"]
    enroll_split = next((s for s in possible
                         if os.path.exists(f"{features_root}/{s}/embeddings.npy")), None)
    if enroll_split is None:
        print(f"  ⚠ No se encontró split de enrolamiento en {features_root}")
        return
    test_split = enroll_split.replace("enroll", "test")

    enroll_embs, enroll_classes = load_split(features_root, enroll_split)
    test_embs,   test_classes   = load_split(features_root, test_split)

    print(f"  ✓ Enrolamiento: {enroll_embs.shape} | Test: {test_embs.shape}")
    print(f"  Token: '{token}'")

    # ── Construir templates de enrolamiento (media por clase) ──────────────
    print("\n🔐 Generando templates cancelables de enrolamiento...")
    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    enroll_templates = {}
    for cls, embs in enroll_by_class.items():
        enroll_templates[cls] = enroll_template(embs, cls, token)

    enrolled_classes = sorted(enroll_templates.keys())
    print(f"  ✓ {len(enroll_templates)} templates de enrolamiento generados")

    rng = np.random.default_rng(42)

    # ── Evaluación con todos los bits (binarización simple) ───────────────
    print("\n📏 [1/2] BER con binarización simple (1024 bits)...")
    simple_genuine  = []
    simple_impostor = []

    for test_emb, test_cls in zip(test_embs, test_classes):
        if test_cls not in enroll_templates:
            continue
        ver_t = verify_template(test_emb, test_cls, token)
        enr_t = enroll_templates[test_cls]
        simple_genuine.append(hamming_distance(ver_t, enr_t))

        other_classes = [c for c in enrolled_classes if c != test_cls]
        if not other_classes:
            continue
        for imp_cls in rng.choice(other_classes, size=min(10, len(other_classes)), replace=False):
            simple_impostor.append(hamming_distance(ver_t, enroll_templates[imp_cls]))

    simple_genuine  = np.array(simple_genuine)
    simple_impostor = np.array(simple_impostor)

    # ── Evaluación con selección de bits fiables ──────────────────────────
    print("\n📏 [2/2] BER con selección de bits fiables...")

    # Construir templates fiables para distintos n_bits
    n_bits_options = [512, 256, 128]
    reliable_results = {}

    for n_bits in n_bits_options:
        print(f"  Evaluando n_bits={n_bits}...")

        # Templates de enrolamiento fiables
        reliable_enroll = {}
        reliable_indices = {}
        for cls, embs in enroll_by_class.items():
            tmpl, idx = enroll_template_reliable(embs, cls, token, n_bits)
            reliable_enroll[cls]  = tmpl
            reliable_indices[cls] = idx

        # BER genuina e impostora
        gen_bers = []
        imp_bers = []
        rng2 = np.random.default_rng(42)

        for test_emb, test_cls in zip(test_embs, test_classes):
            if test_cls not in reliable_enroll:
                continue
            ver_t = verify_template_reliable(
                test_emb, test_cls, token, reliable_indices[test_cls])
            enr_t = reliable_enroll[test_cls]
            gen_bers.append(hamming_distance(ver_t, enr_t))

            other_classes = [c for c in enrolled_classes if c != test_cls]
            if not other_classes:
                continue
            for imp_cls in rng2.choice(other_classes, size=min(10, len(other_classes)), replace=False):
                imp_bers.append(hamming_distance(
                    ver_t, reliable_enroll[imp_cls]))

        reliable_results[n_bits] = {
            "genuine":  np.array(gen_bers),
            "impostor": np.array(imp_bers),
        }

    # ── Imprimir resultados ────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  RESULTADOS — BER por estrategia de binarización")
    print(f"{'═'*64}")
    print(f"  {'Estrategia':<22} {'BER gen μ':>10} {'BER gen P99':>12} "
          f"{'Bits error P99':>14} {'BER imp μ':>10}")
    print(f"{'─'*64}")

    # Binarización simple
    p99_simple = np.percentile(simple_genuine, 99)
    print(f"  {'Simple (1024 bits)':<22} "
          f"{simple_genuine.mean():>10.4f} "
          f"{p99_simple:>12.4f} "
          f"{p99_simple*1024:>14.0f} "
          f"{simple_impostor.mean():>10.4f}")

    # Selección de bits fiables
    for n_bits in n_bits_options:
        r   = reliable_results[n_bits]
        p99 = np.percentile(r["genuine"], 99)
        print(f"  {f'Fiables ({n_bits} bits)':<22} "
              f"{r['genuine'].mean():>10.4f} "
              f"{p99:>12.4f} "
              f"{p99*n_bits:>14.0f} "
              f"{r['impostor'].mean():>10.4f}")

    print(f"{'═'*64}")
    print(f"  → Elige el n_bits con menor 'Bits error P99' manejable para RS")
    print(f"{'═'*64}")

    # ── Guardar estadísticas ───────────────────────────────────────────────
    import json as json_mod
    stats = {
        "run_name": run_name,
        "simple_1024": {
            "genuine_ber_mean": round(float(simple_genuine.mean()), 6),
            "genuine_ber_p99":  round(float(p99_simple), 6),
            "bits_error_p99":   int(np.ceil(p99_simple * 1024)),
            "impostor_ber_mean":round(float(simple_impostor.mean()), 6),
        },
        "reliable_bits": {}
    }
    for n_bits in n_bits_options:
        r   = reliable_results[n_bits]
        p99 = np.percentile(r["genuine"], 99)
        stats["reliable_bits"][str(n_bits)] = {
            "genuine_ber_mean": round(float(r["genuine"].mean()), 6),
            "genuine_ber_p99":  round(float(p99), 6),
            "bits_error_p99":   int(np.ceil(p99 * n_bits)),
            "impostor_ber_mean":round(float(r["impostor"].mean()), 6),
        }

    out_path = Path(results_dir) / "cancelable_ber_stats.json"
    with open(str(out_path), "w") as f:
        json_mod.dump(stats, f, indent=2)
    print(f"\n  ✓ Estadísticas guardadas en {out_path}")

    # ── Figura ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BER genuina vs impostora por estrategia de binarización", fontsize=13)

    configs = [
        ("Simple (1024 bits)", simple_genuine, simple_impostor, 1024),
        ("Fiables 512 bits",   reliable_results[512]["genuine"],
                               reliable_results[512]["impostor"], 512),
        ("Fiables 256 bits",   reliable_results[256]["genuine"],
                               reliable_results[256]["impostor"], 256),
        ("Fiables 128 bits",   reliable_results[128]["genuine"],
                               reliable_results[128]["impostor"], 128),
    ]

    for ax, (title, gen, imp, nbits) in zip(axes.flat, configs):
        p99 = np.percentile(gen, 99)
        ax.hist(imp, bins=40, alpha=0.6, color='red',
                label=f"Impostores (μ={imp.mean():.3f})", density=True)
        ax.hist(gen, bins=40, alpha=0.6, color='blue',
                label=f"Genuinos (μ={gen.mean():.3f})", density=True)
        ax.axvline(p99, color='navy', linestyle='--',
                   label=f"P99={p99:.3f} ({p99*nbits:.0f} bits)")
        ax.set_title(title)
        ax.set_xlabel("BER")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = str(Path(results_dir) / "cancelable_ber_distribution.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"  ✓ Figura guardada en {out_fig}")
    print("\n✅ Evaluación completada.\n")

    return stats


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Transformación cancelable de iris")
    parser.add_argument("--run_name",   type=str, default="exp07",
                        help="Nombre del experimento")
    parser.add_argument("--token_path", type=str, default="data/system_token.key",
                        help="Ruta al fichero con el token del sistema")
    args = parser.parse_args()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)
    results_dir   = str(Path(config["paths"]["results"])           / args.run_name)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("\n🔑 Cargando token del sistema...")
    token = load_or_create_system_token(args.token_path)

    evaluate(args.run_name, token, results_dir, features_root)


if __name__ == "__main__":
    main()
