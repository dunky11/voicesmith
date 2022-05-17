from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
from pathlib import Path
from voice_smith.utils.audio import safe_load
from typing import List, Callable, Optional, Union
from voice_smith.utils.tools import iter_logger


def get_txt_from_file(src: str) -> Union[str, None]:
    if not Path(src).exists():
        print(f"Text file {src} doesn't exist, skipping ...")
        return None
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_txt_from_files(
    db_id: int,
    table_name: str,
    txt_paths: List[str],
    get_logger: Optional[Callable],
    log_every: int = 200,
) -> List[str]:
    def callback(index: int):
        if index % log_every == 0:
            logger = get_logger()
            progress = index / len(txt_paths)
            logger.query(
                f"UPDATE {table_name} SET get_txt_progress=? WHERE id=?",
                [progress, db_id],
            )

    print("Fetching text from files ...")
    texts = Parallel(n_jobs=max(1, mp.cpu_count() - 1))(
        delayed(get_txt_from_file)(file_path)
        for file_path in iter_logger(txt_paths, cb=callback)
    )
    logger = get_logger()
    logger.query(
        f"UPDATE {table_name} SET get_txt_progress=? WHERE id=?",
        [1.0, db_id],
    )
    return texts
