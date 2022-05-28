from voice_smith.utils.tools import warnings_to_stdout

warnings_to_stdout()
from pathlib import Path
import shutil
import argparse
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.preprocessing.text_normalization import text_normalize


def continue_text_normalization_run(
    text_normalization_run_id: int,
    db_path: str,
    text_normalization_runs_path: str,
    assets_path: str,
    user_data_path: str,
):
    con = get_con(db_path)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
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

            row = cur.execute(
                "SELECT language FROM text_normalization_run WHERE ID=?",
                (text_normalization_run_id,),
            ).fetchone()
            lang = row[0]
            cur.execute(
                "DELETE FROM text_normalization_sample WHERE text_normalization_run_id = ?",
                (text_normalization_run_id,),
            )
            con.commit()
            id_text_pairs = []
            for (sample_id, text) in cur.execute(
                """
                SELECT sample.ID AS sampleID, sample.text FROM sample
                INNER JOIN speaker ON sample.speaker_id = speaker.ID
                INNER JOIN dataset on speaker.dataset_id = dataset.ID
                INNER JOIN text_normalization_run ON text_normalization_run.dataset_id = dataset.ID
                WHERE text_normalization_run.ID = ?
                """,
                (text_normalization_run_id,),
            ).fetchall():
                id_text_pairs.append((sample_id, text))

            def callback(progress: float):
                progress = progress * 0.9
                cur.execute(
                    "UPDATE text_normalization_run SET text_normalization_progress=1.0 WHERE ID=?",
                    (text_normalization_run_id,),
                )
                con.commit()

            normalizations = text_normalize(
                id_text_pairs=id_text_pairs,
                assets_path=assets_path,
                lang=lang,
                progress_cb=callback,
            )

            for (sample_id, text_in, text_out, reason) in normalizations:
                cur.execute(
                    """
                    INSERT INTO text_normalization_sample (old_text, new_text, reason, sample_id, text_normalization_run_id) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (text_in, text_out, reason, sample_id, text_normalization_run_id),
                )

            cur.execute(
                "UPDATE text_normalization_run SET stage='choose_samples', text_normalization_progress=1.0 WHERE ID=?",
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
    parser.add_argument("--assets_path", type=str, required=True)
    args = parser.parse_args()

    continue_text_normalization_run(
        text_normalization_run_id=args.text_normalization_run_id,
        db_path=args.db_path,
        text_normalization_runs_path=args.text_normalization_runs_path,
        user_data_path=args.user_data_path,
        assets_path=args.assets_path,
    )
