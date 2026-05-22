from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./mri_memory.db"

# SQLite needs this flag for FastAPI multithreaded request handling.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)


class Base(DeclarativeBase):
    """Shared declarative base for ORM models."""


def get_db():
    """FastAPI dependency for per-request DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
