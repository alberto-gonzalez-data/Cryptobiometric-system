"""
07_fuzzy_commitment.py
Librería de fuzzy commitment para el sistema criptobiométrico de iris.

Implementa el esquema de compromiso difuso descrito en el PDF:
  1. Generación de secreto aleatorio
  2. Codificación Reed-Solomon (tolerancia a errores)
  3. Generación del commitment (XOR template cancelable + codeword RS)
  4. Verificación Argon2 del secreto recuperado

Este módulo es una librería importada por:
  - 09_crypto_pipeline.py  → pipeline completo con base de datos
  - biometric_server.py    → demo web, incluida la identificación 1:N
  - scripts eval_*.py      → evaluación del sistema

Limitación conocida:
  Reed-Solomon trabaja a nivel de byte, mientras que los errores biométricos
  se distribuyen a nivel de bit, lo que limita la tasa de aceptación genuina.
  Resultados con exp08 (256 bits, 16 B de paridad, 5+3 imágenes):
  GAR = 85.09 % y FAR = 0.0030 % (ver results/exp08/eval_biometric_report.pdf).
  Como trabajo futuro se propone BCH para corrección directa a nivel de bit.

Uso como librería:
    import importlib
    _fc = importlib.import_module("07_fuzzy_commitment")
    record = _fc.enroll_user(embeddings, user_id, n_bits=256, n_parity=16)
    result = _fc.authenticate_user(embeddings, user_id, record)
"""

import os
import sys
import secrets
import numpy as np
from pathlib import Path
from reedsolo import RSCodec, ReedSolomonError
from argon2 import PasswordHasher

sys.path.insert(0, str(Path(__file__).parent))
import importlib

_cancelable              = importlib.import_module("06_cancelable")
enroll_template_reliable = _cancelable.enroll_template_reliable
verify_template_reliable = _cancelable.verify_template_reliable
hamming_distance         = _cancelable.hamming_distance


# ─────────────────────────────────────────────
# 1. ARGON2
# ─────────────────────────────────────────────

def create_hasher():
    """
    Crea un PasswordHasher con los parámetros recomendados en el PDF.
    memory_cost=8192 KB, time_cost=3, parallelism=2.
    Diseñado para ser deliberadamente lento y resistente a ataques GPU.
    """
    return PasswordHasher(memory_cost=8192, time_cost=3, parallelism=2)


def argon2_hash(secret: bytes) -> str:
    """Hashea el secreto con Argon2. Devuelve el hash como string."""
    return create_hasher().hash(secret.hex())


def argon2_verify(hash_str: str, secret: bytes) -> bool:
    """Verifica si el secreto coincide con el hash Argon2 almacenado."""
    try:
        create_hasher().verify(hash_str, secret.hex())
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# 2. FUZZY COMMITMENT
# ─────────────────────────────────────────────

def generate_commitment(cancelable_template: np.ndarray,
                        n_parity: int,
                        rng=None) -> dict:
    """
    Genera el commitment criptográfico a partir del template cancelable.

    Pipeline:
      1. Genera secreto aleatorio s de (n_bytes - n_parity) bytes
      2. Codifica s con Reed-Solomon -> codeword c de n_bytes bytes
      3. Convierte template binario a bytes
      4. Commitment delta = template_bytes XOR codeword
      5. Hash h(s) = Argon2(s)

    El secreto nunca se almacena -- solo el commitment y el hash.
    Sin el template correcto no es posible recuperar el secreto.

    Args:
        cancelable_template: vector binario cancelable, shape (n_bits,)
                             donde n_bits debe ser divisible por 8
        n_parity:            bytes de paridad RS
                             (corrige hasta n_parity//2 errores de byte)
        rng:                 generador numpy opcional. Por defecto (None) el
                             secreto sale de os.urandom, que es lo que usa el
                             sistema real. La evaluacion pasa un generador con
                             semilla para que los resultados sean reproducibles.

    Returns:
        dict con commitment (hex), hash_secret, secret_len, n_parity, n_bits
    """
    n_bits       = len(cancelable_template)
    n_bytes      = n_bits // 8
    secret_bytes = n_bytes - n_parity

    if secret_bytes <= 0:
        raise ValueError(
            f"n_parity={n_parity} deja {secret_bytes} bytes para el secreto. "
            f"Reduce n_parity o aumenta n_bits.")

    secret   = os.urandom(secret_bytes) if rng is None else rng.bytes(secret_bytes)
    rsc      = RSCodec(n_parity)
    codeword = bytes(rsc.encode(bytearray(secret)))

    template_bytes = np.packbits(cancelable_template).tobytes()
    commitment     = bytes(a ^ b for a, b in zip(template_bytes, codeword))
    hash_secret    = argon2_hash(secret)

    return {
        "commitment":  commitment.hex(),
        "hash_secret": hash_secret,
        "secret_len":  secret_bytes,
        "n_parity":    n_parity,
        "n_bits":      n_bits,
    }


def verify_commitment(cancelable_template: np.ndarray,
                      commitment_data: dict) -> tuple:
    """
    Intenta recuperar el secreto del commitment usando el template de verificacion.

    Pipeline:
      1. Convierte template a bytes
      2. XOR con commitment -> codeword ruidosa
         (codeword_ruidosa = codeword XOR errores_entre_templates)
      3. Reed-Solomon corrige errores -> secreto recuperado (o fallo)
      4. Argon2 verifica el secreto recuperado

    Args:
        cancelable_template: template cancelable de verificacion, shape (n_bits,)
        commitment_data:     dict generado por generate_commitment()

    Returns:
        (success: bool, secret_recovered: bytes or None, info: str)
    """
    n_parity    = commitment_data["n_parity"]
    commitment  = bytes.fromhex(commitment_data["commitment"])
    hash_secret = commitment_data["hash_secret"]

    template_bytes = np.packbits(cancelable_template).tobytes()
    codeword_noisy = bytes(a ^ b for a, b in zip(template_bytes, commitment))

    rsc = RSCodec(n_parity)
    try:
        decoded = rsc.decode(bytearray(codeword_noisy))
        secret_recovered = bytes(
            decoded[0] if isinstance(decoded, tuple) else decoded)

        if argon2_verify(hash_secret, secret_recovered):
            return True, secret_recovered, "RS decodificado y Argon2 verificado"
        else:
            return False, None, "RS decodificado pero Argon2 no coincide"

    except ReedSolomonError as e:
        return False, None, f"RS fallo: {str(e)}"
    except Exception as e:
        return False, None, f"Error inesperado: {str(e)}"


# ─────────────────────────────────────────────
# 3. PIPELINES DE ENROLAMIENTO Y AUTENTICACION
# ─────────────────────────────────────────────

def enroll_user(embeddings: list,
                user_id: str,
                n_bits: int = 256,
                n_parity: int = 16,
                rng=None) -> dict:
    """
    Pipeline completo de enrolamiento para un usuario.

    Genera un transform_seed individual aleatorio de 256 bits para
    garantizar la revocabilidad independiente por usuario.

    Pipeline:
      1. Genera transform_seed individual (256 bits aleatorios)
      2. Genera template cancelable + indices de bits fiables
      3. Genera commitment criptografico (secreto + RS + XOR + Argon2)

    Args:
        embeddings: lista de embeddings de enrolamiento (rec. 5)
        user_id:    identificador unico del usuario
        n_bits:     bits fiables del template (default 256)
        n_parity:   bytes de paridad RS (default 16)
        rng:        generador numpy opcional (solo para evaluacion). Si es None,
                    la semilla y el secreto salen de la entropia del sistema
                    operativo (secrets / os.urandom), como en produccion.
                    Si se pasa, ambos se extraen de el, en este orden: primero
                    la semilla de transformacion y despues el secreto.

    Returns:
        dict con todos los datos necesarios para autenticacion futura:
        commitment, hash_secret, reliable_indices, transform_seed,
        n_bits, n_parity, n_enroll_images
    """
    transform_seed  = (secrets.token_hex(32) if rng is None
                       else rng.bytes(32).hex())
    cancelable, indices = enroll_template_reliable(
        embeddings, user_id, transform_seed, n_bits)
    commitment_data = generate_commitment(cancelable, n_parity, rng=rng)

    return {
        "user_id":          user_id,
        "commitment":       commitment_data["commitment"],
        "hash_secret":      commitment_data["hash_secret"],
        "reliable_indices": indices.tolist(),
        "transform_seed":   transform_seed,
        "n_bits":           n_bits,
        "n_parity":         n_parity,
        "n_enroll_images":  len(embeddings),
    }


def authenticate_user(embeddings: list,
                      user_id: str,
                      user_record: dict) -> tuple:
    """
    Pipeline completo de autenticacion para un usuario.

    Pipeline:
      1. Calcula media de embeddings de verificacion
      2. Genera template cancelable con el transform_seed del registro
      3. XOR con commitment -> codeword ruidosa
      4. Reed-Solomon corrige errores -> secreto
      5. Argon2 verifica el secreto

    Args:
        embeddings:  lista de embeddings de verificacion (rec. 3)
        user_id:     identificador del usuario
        user_record: dict devuelto por enroll_user()

    Returns:
        (success: bool, info: str)
    """
    indices        = np.array(user_record["reliable_indices"])
    transform_seed = user_record["transform_seed"]

    mean_emb = np.mean(np.array(embeddings), axis=0)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm

    cancelable_ver = verify_template_reliable(
        mean_emb, user_id, transform_seed, indices)

    commitment_data = {
        "commitment":  user_record["commitment"],
        "hash_secret": user_record["hash_secret"],
        "n_parity":    user_record["n_parity"],
    }
    success, _, info = verify_commitment(cancelable_ver, commitment_data)

    return success, info
