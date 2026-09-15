from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rag import config


def _make_engine():
    connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}
    return create_engine(config.DATABASE_URL, connect_args=connect_args)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from rag import db_models  # noqa: F401  (registers models on Base before create_all)

    Base.metadata.create_all(bind=engine)
