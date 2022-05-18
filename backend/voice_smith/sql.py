import sqlite3
import pathlib
from pathlib import Path
    
def get_con(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    return con
