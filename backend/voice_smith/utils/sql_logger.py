from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Union
from voice_smith.utils.audio import save_audio
from voice_smith.utils.loggers import Logger
 

class SQLLogger(Logger):
    def __init__(self, training_run_id: int, con, cursor, out_dir: str, stage: str):
        super().__init__()
        self.training_run_id = training_run_id
        self.con = con
        self.cur = cursor
        self.out_dir = Path(out_dir)
        self.stage = stage

    def log_image(self, name: str, image: np.ndarray, step: int) -> None:
        image = self.map_image_color(image)
        out_dir = self.out_dir / "image_logs" / name
        out_dir.mkdir(exist_ok=True, parents=True)
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
        pil_img.save(str(out_dir / f"{step}.png"))
        self.cur.execute(
            "INSERT INTO image_statistic(name, step, stage, training_run_id) VALUES(?, ?, ?, ?)",
            [name, step, self.stage, self.training_run_id],
        )
        self.con.commit()

    def log_graph(self, name: str, value: float, step: int):
        self.cur.execute(
            "INSERT INTO graph_statistic(name, step, stage, value, training_run_id) VALUES(?, ?, ?, ?, ?)",
            [name, step, self.stage, value, self.training_run_id],
        )
        self.con.commit()

    def log_audio(self, name: str, audio: np.ndarray, step: int, sr: int):
        out_dir = self.out_dir / "audio_logs" / name
        out_dir.mkdir(exist_ok=True, parents=True)
        save_audio(str(out_dir / f"{step}.flac"), torch.FloatTensor(audio), sr)
        self.cur.execute(
            "INSERT INTO audio_statistic(name, step, stage, training_run_id) VALUES(?, ?, ?, ?)",
            [name, step, self.stage, self.training_run_id],
        )
        self.con.commit()

    def query(self, query: str, args: List[Union[int, str]]) -> None:
        self.cur.execute(query, args)
        self.con.commit()
 