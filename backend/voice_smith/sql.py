import sqlite3
import pathlib
from pathlib import Path
import os


def get_con(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    return con


def save_current_pid(con: sqlite3.Connection, cur: sqlite3.Cursor):
    cur.execute("UPDATE settings SET pid=?", (os.getpid(), ))
    con.commit()
