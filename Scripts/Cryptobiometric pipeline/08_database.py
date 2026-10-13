"""
08_database.py
Base de datos SQLite para el sistema criptobiométrico de iris.

Implementa la capa de persistencia usando SQLAlchemy + SQLite,
tal como recomienda la sección 8 del PDF.

Schema de la tabla 'users':
  user_id:          identificador único del usuario
  commitment:       fuzzy commitment (hex, 32 bytes)
  hash_secret:      hash Argon2 del secreto
  reliable_indices: índices de los 256 bits fiables (JSON)
  transform_seed:   seed individual para la transformación cancelable (hex 64 chars)
  n_bits:           número de bits del template
  n_parity:         bytes de paridad Reed-Solomon
  n_enroll_images:  número de imágenes usadas en el enrolamiento
  created_at:       timestamp del enrolamiento
  extra_metadata:   información adicional opcional (JSON)

Uso:
    from database import init_db, get_session, add_user, get_user, list_users, delete_user
"""

from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path

Base = declarative_base()


# ─────────────────────────────────────────────
# 1. SCHEMA
# ─────────────────────────────────────────────

class UserRecord(Base):
    """
    Registro de un usuario en el sistema criptobiométrico.
    Almacena todo lo necesario para verificar al usuario
    sin guardar ni el iris ni el secreto.
    """
    __tablename__ = "users"

    user_id          = Column(String,  primary_key=True)
    commitment       = Column(String,  nullable=False)   # hex, 32 bytes
    hash_secret      = Column(String,  nullable=False)   # Argon2 hash
    reliable_indices = Column(Text,    nullable=False)   # JSON array de 256 ints
    transform_seed   = Column(String,  nullable=False)   # hex 64 chars (256 bits)
    n_bits           = Column(Integer, nullable=False)   # bits del template
    n_parity         = Column(Integer, nullable=False)   # bytes de paridad RS
    n_enroll_images  = Column(Integer, nullable=False)   # imágenes de enrolamiento
    created_at       = Column(String,  nullable=False)   # timestamp ISO
    extra_metadata   = Column(Text,    nullable=True)    # JSON opcional

    def __repr__(self):
        return (f"<UserRecord user_id={self.user_id} "
                f"n_bits={self.n_bits} created_at={self.created_at}>")


# ─────────────────────────────────────────────
# 2. CONEXIÓN
# ─────────────────────────────────────────────

def get_engine(db_path: str):
    """Crea el motor SQLAlchemy para la base de datos SQLite."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(db_path: str):
    """Inicializa la base de datos creando las tablas si no existen."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    print(f"  ✓ Base de datos inicializada en {db_path}")
    return engine


def get_session(engine):
    """Crea una sesión de base de datos."""
    Session = sessionmaker(bind=engine)
    return Session()


# ─────────────────────────────────────────────
# 3. CRUD
# ─────────────────────────────────────────────

def add_user(session, record: dict) -> bool:
    """
    Añade un usuario a la base de datos.
    Si ya existe, actualiza su registro (re-enrolamiento).
    Devuelve True si es nuevo, False si es actualización.
    """
    existing = session.query(UserRecord).filter_by(
        user_id=record["user_id"]).first()

    if existing:
        for key, value in record.items():
            setattr(existing, key, value)
        session.commit()
        return False
    else:
        session.add(UserRecord(**record))
        session.commit()
        return True


def get_user(session, user_id: str):
    """Recupera el registro de un usuario por su user_id."""
    return session.query(UserRecord).filter_by(user_id=user_id).first()


def list_users(session) -> list:
    """Lista todos los usuarios registrados."""
    return session.query(UserRecord).all()


def delete_user(session, user_id: str) -> bool:
    """Elimina un usuario de la base de datos. Devuelve True si existía."""
    user = session.query(UserRecord).filter_by(user_id=user_id).first()
    if user:
        session.delete(user)
        session.commit()
        return True
    return False


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Base de datos SQLite del sistema biométrico")
    parser.add_argument("--db_path", type=str, default="data/biometric.db")
    parser.add_argument("--list",    action="store_true",
                        help="Listar usuarios registrados")
    args = parser.parse_args()

    engine  = init_db(args.db_path)
    session = get_session(engine)

    if args.list:
        users = list_users(session)
        if users:
            print(f"\n  {'user_id':<35} {'n_bits':>6} {'created_at'}")
            print(f"  {'─'*65}")
            for u in users:
                print(f"  {u.user_id:<35} {u.n_bits:>6} {u.created_at}")
            print(f"\n  Total: {len(users)} usuarios")
        else:
            print("  No hay usuarios registrados.")

    session.close()
