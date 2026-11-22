import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT


def home():
    return Path(os.getenv("MODELLAB_DATA_DIR", str(PROJECT / ".runtime"))).resolve()


def uid():
    return uuid.uuid4().hex


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect():
    home().mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(
        home() / "modellab.sqlite", timeout=15, factory=ClosingConnection
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=15000")
    return db


@contextmanager
def transaction():
    db = connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        migrations_dir = PROJECT / "backend" / "migrations"
        for sql_file in sorted(migrations_dir.glob("*.sql")):
            db.executescript(sql_file.read_text(encoding="utf-8"))
