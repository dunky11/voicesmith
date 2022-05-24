from pathlib import Path
import shutil
import argparse
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con
from voice_smith.docker.api import reload_docker, text_normalize


def continue_text_normalization_run(
    text_normalization_run_id: int,
    db_path: str,
    text_normalization_runs_path: str,
    user_data_path: str,
):
    con = get_con(db_path)
    cur = con.cursor()
    data_path = Path(text_normalization_runs_path) / str(text_normalization_run_id)
    stage = None

    while (
        stage != "choose_samples" and stage != "apply_changes" and stage != "finished"
    ):
        row = cur.execute(
            "SELECT stage FROM text_normalization_run WHERE ID=?",
            (text_normalization_run_id,),
        ).fetchone()
        stage = row[0]
        if stage == "not_started":
            if data_path.exists():
                shutil.rmtree(data_path)
            (data_path / "logs").mkdir(exist_ok=True, parents=True)
            cur.execute(
                "UPDATE text_normalization_run SET stage='text_normalization' WHERE ID=?",
                (text_normalization_run_id,),
            )
            con.commit()

        elif stage == "text_normalization":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            print("Normalizing Text ...\n")
            container = reload_docker(user_data_path=user_data_path, db_path=db_path)

            row = cur.execute(
                "SELECT language FROM text_normalization_run WHERE ID=?",
                (text_normalization_run_id,),
            ).fetchone()
            lang = row[0]
            text_normalize(container, str(text_normalization_run_id), lang=lang)
            cur.execute(
                "UPDATE text_normalization_run SET stage='choose_samples' WHERE ID=?",
                (text_normalization_run_id,),
            )
            con.commit()

        elif stage == "choose_samples":
            pass

        elif stage == "apply_changes":
            pass

        elif stage == "finished":
            pass

        else:
            raise Exception(f"Stage stage '{stage}' is not a valid stage ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_normalization_run_id", type=int, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--text_normalization_runs_path", type=str, required=True)
    parser.add_argument("--user_data_path", type=str, required=True)
    args = parser.parse_args()

    continue_text_normalization_run(
        text_normalization_run_id=args.text_normalization_run_id,
        db_path=args.db_path,
        text_normalization_runs_path=args.text_normalization_runs_path,
        user_data_path=args.user_data_path,
    )
