import sqlite3
from typing import Optional, Callable, List, Tuple, Any, Dict


class StageRunner:
    def __init__(
        self,
        cur: sqlite3.Cursor,
        con: sqlite3.Connection,
        get_stage_name: Callable[[Any], str],
        stages: List[Tuple[str, Callable[[Any, ...], bool]]],
        before_stage: Optional[Callable[[Any, ...], None]] = None,
        after_stage: Optional[Callable[[Any], None]] = None,
        before_run: Optional[Callable[[Any], None]] = None,
        after_run: Optional[Callable[[Any], None]] = None,
    ):
        self.cur = cur
        self.con = con
        self.get_stage_name = get_stage_name
        self.stages = stages
        self.before_stage = before_stage
        self.after_stage = after_stage
        self.before_run = before_run
        self.after_run = after_run

    def run(self, **kwargs):
        finished = False
        if self.before_run is not None:
            self.before_run(**kwargs)
        while not finished:
            stage_name = self.get_stage_name(**kwargs)
            for n, stage in self.stages:
                if n == stage_name:
                    if self.before_stage is not None:
                        self.before_stage(**{**kwargs, "stage_name": stage_name})
                    finished = stage(**kwargs)
                    if self.after_stage is not None:
                        self.after_stage(**{**kwargs, "stage_name": stage_name})
                    break
        if self.after_run is not None:
            self.after_run(**kwargs)

