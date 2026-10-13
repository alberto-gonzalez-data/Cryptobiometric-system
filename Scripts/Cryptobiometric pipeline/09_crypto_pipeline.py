"""
09_crypto_pipeline.py
Pipeline completo del sistema criptobiométrico de iris.

Integra todos los módulos del sistema:
  - Transformación cancelable (06_cancelable.py)
  - Fuzzy commitment con Reed-Solomon (07_fuzzy_commitment.py)
  - Protección Argon2
  - Base de datos SQLite (08_database.py)

Proporciona:
  - enroll_user():      pipeline completo de enrolamiento
  - authenticate_user(): pipeline completo de autenticación
  - Demo con embeddings de exp08

Uso:
    # Demo completa
    python scripts/09_crypto_pipeline.py --run_name exp08 --demo

    # Enrolar un usuario específico (requiere embeddings)
    python scripts/09_crypto_pipeline.py --run_name exp08 --enroll user_id

    # Autenticar un usuario específico
    python scripts/09_crypto_pipeline.py --run_name exp08 --authenticate user_id
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import yaml
import random

sys.path.insert(0, str(Path(__file__).parent))
import importlib

_fc               = importlib.import_module("07_fuzzy_commitment")
enroll_user_fc    = _fc.enroll_user
authenticate_user_fc = _fc.authenticate_user

_database   = importlib.import_module("08_database")
init_db     = _database.init_db
get_session = _database.get_session
add_user    = _database.add_user
get_user    = _database.get_user
list_users  = _database.list_users


# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. PIPELINE DE ENROLAMIENTO
# ─────────────────────────────────────────────
# Argon2, RS commitment y pipelines importados de 07_fuzzy_commitment

def enroll_user(session,
                user_id: str,
                embeddings: list,
                n_bits: int = 256,
                n_parity: int = 16,
                extra_metadata: dict = None) -> dict:
    """
    Pipeline completo de enrolamiento de un usuario.

    Delega la lógica criptográfica en 07_fuzzy_commitment.enroll_user
    y añade la capa de persistencia en la base de datos SQLite.

    Args:
        session:        sesión de base de datos
        user_id:        identificador único del usuario
        embeddings:     lista de embeddings de enrolamiento (rec. 5)
        n_bits:         bits fiables del template (default 256)
        n_parity:       bytes de paridad RS (default 16)
        extra_metadata: información adicional opcional

    Returns:
        dict con resultado del enrolamiento
    """
    # Pipeline criptográfico (07_fuzzy_commitment)
    crypto_record = enroll_user_fc(
        embeddings, user_id, n_bits=n_bits, n_parity=n_parity)

    # Guardar en base de datos
    db_record = {
        "user_id":          user_id,
        "commitment":       crypto_record["commitment"],
        "hash_secret":      crypto_record["hash_secret"],
        "reliable_indices": json.dumps(crypto_record["reliable_indices"]),
        "transform_seed":   crypto_record["transform_seed"],
        "n_bits":           n_bits,
        "n_parity":         n_parity,
        "n_enroll_images":  len(embeddings),
        "created_at":       datetime.now().isoformat(),
        "extra_metadata":   json.dumps(extra_metadata) if extra_metadata else None,
    }

    is_new = add_user(session, db_record)
    action = "registrado" if is_new else "actualizado"

    return {
        "success":     True,
        "user_id":     user_id,
        "action":      action,
        "n_images":    len(embeddings),
        "secret_bits": (n_bits // 8 - n_parity) * 8,
        "n_parity":    n_parity,
    }


# ─────────────────────────────────────────────
# 4. PIPELINE DE AUTENTICACIÓN
# ─────────────────────────────────────────────

def authenticate_user(session,
                      user_id: str,
                      embeddings: list) -> dict:
    """
    Pipeline completo de autenticación de un usuario.

    Recupera el registro de la base de datos y delega la verificación
    criptográfica en 07_fuzzy_commitment.authenticate_user.

    Args:
        session:    sesión de base de datos
        user_id:    identificador del usuario a autenticar
        embeddings: lista de embeddings de verificación (rec. 3)

    Returns:
        dict con resultado de la autenticación
    """
    # Recuperar registro de la base de datos
    user = get_user(session, user_id)
    if user is None:
        return {
            "success": False,
            "user_id": user_id,
            "reason":  "Usuario no registrado"
        }

    # Construir user_record en el formato de 07_fuzzy_commitment
    user_record = {
        "commitment":       user.commitment,
        "hash_secret":      user.hash_secret,
        "reliable_indices": json.loads(user.reliable_indices),
        "transform_seed":   user.transform_seed,
        "n_parity":         user.n_parity,
        "n_bits":           user.n_bits,
    }

    # Autenticación criptográfica (07_fuzzy_commitment)
    success, info = authenticate_user_fc(embeddings, user_id, user_record)

    return {
        "success": success,
        "user_id": user_id,
        "reason":  info,
    }


# ─────────────────────────────────────────────
# 5. DEMO
# ─────────────────────────────────────────────

def run_demo(run_name: str, db_path: str, features_root: str,
             n_bits: int, n_parity: int, n_enroll: int, n_verify: int):
    """
    Demo completa del sistema usando embeddings guardados de exp08.
    Enrola y autentica usuarios mostrando el flujo completo.
    """
    print(f"\n{'═'*60}")
    print(f"  DEMO — Sistema Criptobiométrico Completo")
    print(f"{'═'*60}")
    print(f"  Experimento : {run_name}")
    print(f"  n_bits      : {n_bits} | n_parity: {n_parity}")
    print(f"  n_enroll    : {n_enroll} imágenes | n_verify: {n_verify} imágenes")
    print(f"  Secreto     : {n_bits//8 - n_parity} bytes = {(n_bits//8-n_parity)*8} bits")

    # Inicializar base de datos
    engine  = init_db(db_path)
    session = get_session(engine)

    # Cargar embeddings
    def load_split(split_name):
        split_dir = Path(features_root) / split_name
        embs      = np.load(str(split_dir / "embeddings.npy"))
        classes   = json.load(open(str(split_dir / "class_strings.json")))
        return embs, classes

    possible     = ["enroll_joint", "enroll_phase2", "enroll_phase1", "enroll"]
    enroll_split = next((s for s in possible
                         if (Path(features_root)/s/"embeddings.npy").exists()), None)
    test_split   = enroll_split.replace("enroll", "test")

    enroll_embs, enroll_classes = load_split(enroll_split)
    test_embs,   test_classes   = load_split(test_split)

    enroll_by_class = defaultdict(list)
    for emb, cls in zip(enroll_embs, enroll_classes):
        enroll_by_class[cls].append(emb)

    test_by_class = defaultdict(list)
    for emb, cls in zip(test_embs, test_classes):
        test_by_class[cls].append(emb)

    # ── Enrolamiento ───────────────────────────────────────────────────────
    print(f"\n🔐 Enrollando usuarios (máx. {n_enroll} imágenes)...")
    n_enrolled = 0
    for cls, embs in enroll_by_class.items():
        try:
            enroll_user(session, cls, embs[:n_enroll],
                        n_bits=n_bits, n_parity=n_parity)
            n_enrolled += 1
        except Exception as e:
            print(f"  ⚠ Error enrollando {cls}: {e}")

    print(f"  ✓ {n_enrolled} usuarios enrollados")
    print(f"  ✓ Usuarios en BD: {len(list_users(session))}")

    # ── Autenticación genuina (muestra de 10 usuarios) ─────────────────────
    print(f"\n🔑 Verificación genuina (muestra de 10 usuarios)...")
    print(f"{'─'*60}")
    print(f"  {'Usuario':<30} {'Resultado':<20} {'Motivo'}")
    print(f"{'─'*60}")

    enrolled_ids     = {u.user_id for u in list_users(session)}
    all_test_classes = [c for c in test_by_class if c in enrolled_ids]
    sample_classes   = random.sample(all_test_classes,
                                     min(10, len(all_test_classes)))
    genuine_success = 0

    for cls in sample_classes:
        test_arr = np.array(test_by_class[cls])
        window   = list(test_arr[:n_verify])
        result   = authenticate_user(session, cls, window)
        status   = "✓ ACCESO CONCEDIDO" if result["success"] else "✗ DENEGADO"
        if result["success"]:
            genuine_success += 1
        print(f"  {cls:<30} {status:<20} {result['reason']}")

    print(f"{'─'*60}")
    print(f"  Éxitos genuinos: {genuine_success}/{len(sample_classes)}")

    # ── Autenticación impostora (5 intentos) ──────────────────────────────
    print(f"\n🚫 Verificación impostora (5 intentos)...")
    print(f"{'─'*60}")
    print(f"  {'Imagen de':<25} {'Contra usuario':<25} {'Resultado'}")
    print(f"{'─'*60}")

    rng              = random.Random(42)
    enrolled_classes = list(enroll_by_class.keys())
    impostor_success = 0

    for _ in range(5):
        query_cls  = rng.choice(sample_classes)
        target_cls = rng.choice([c for c in enrolled_classes if c != query_cls])
        window     = list(np.array(test_by_class[query_cls])[:n_verify])
        result     = authenticate_user(session, target_cls, window)
        status     = "✗ DENEGADO" if not result["success"] else "⚠ ACEPTADO (FAR)"
        if result["success"]:
            impostor_success += 1
        print(f"  {query_cls[:24]:<25} {target_cls[:24]:<25} {status}")

    print(f"{'─'*60}")
    print(f"  Falsas aceptaciones: {impostor_success}/5")
    print(f"\n{'═'*60}")
    print(f"  DEMO COMPLETADA")
    print(f"{'═'*60}\n")

    session.close()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        "Pipeline completo del sistema criptobiométrico")
    parser.add_argument("--run_name", type=str,  default="exp08")
    parser.add_argument("--db_path",  type=str,  default="data/biometric.db")
    parser.add_argument("--n_bits",   type=int,  default=256)
    parser.add_argument("--n_parity", type=int,  default=16)
    parser.add_argument("--n_enroll", type=int,  default=5)
    parser.add_argument("--n_verify", type=int,  default=3)
    parser.add_argument("--demo",     action="store_true",
                        help="Ejecutar demo completa con embeddings guardados")
    args = parser.parse_args()

    config        = load_config("config.yaml")
    features_root = str(Path(config["paths"]["data"]["features"]) / args.run_name)

    if args.demo:
        run_demo(
            run_name=args.run_name,
            db_path=args.db_path,
            features_root=features_root,
            n_bits=args.n_bits,
            n_parity=args.n_parity,
            n_enroll=args.n_enroll,
            n_verify=args.n_verify,
        )
    else:
        print("Usa --demo para ejecutar la demo completa.")


if __name__ == "__main__":
    main()
