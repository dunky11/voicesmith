from pathlib import Path
import shutil
import torch
import numpy as np
import argparse
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.utils.loggers import set_stream_location
from voice_smith.preprocessing.dataset_cleaning import get_issues, load_embeddings
from voice_smith.sql import get_con
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.preprocessing.get_txt_from_files import get_txt_from_files
from voice_smith.preprocessing.gen_speaker_embeddings import (
    gen_file_emeddings,
)


def continue_cleaning_run(
    cleaning_run_id: int,
    db_path: str,
    preprocessing_runs_path: str,
    datasets_path: str
):
    con = get_con(db_path)
    cur = con.cursor()
    data_path = (
        Path(preprocessing_runs_path) / "cleaning_runs" / str(cleaning_run_id)
    )
    dataset_path = Path(datasets_path)
    stage = None

    def get_logger():
        con = get_con(db_path)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=cleaning_run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="cleaning_run",
        )
        return logger

    while stage != "finished" and stage != "choose_samples":
        row = cur.execute(
            "SELECT stage FROM cleaning_run WHERE ID=?",
            (cleaning_run_id,),
        ).fetchone()
        stage = row[0]
        if stage == "not_started":
            if data_path.exists():
                shutil.rmtree(data_path)
            (data_path / "speaker_embeds_per_file").mkdir(exist_ok=True, parents=True)
            (data_path / "logs").mkdir(exist_ok=True, parents=True)
            cur.execute(
                "UPDATE cleaning_run SET stage='gen_file_embeddings' WHERE ID=?",
                (cleaning_run_id,),
            )
            con.commit()

        elif stage == "gen_file_embeddings":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))

            audio_paths = []
            for (audio_path, dataset_id, speaker_id) in cur.execute(
                """
                SELECT sample.audio_path, dataset.ID AS dataset_id, speaker.ID as speaker_id FROM sample
                INNER JOIN speaker ON sample.speaker_id = speaker.ID
                INNER JOIN dataset ON speaker.dataset_id = dataset.ID
                INNER JOIN cleaning_run ON cleaning_run.dataset_id = dataset.ID
                WHERE cleaning_run.ID = ?
                """,
                (cleaning_run_id,),
            ).fetchall():
                full_audio_path = (
                    dataset_path
                    / str(dataset_id)
                    / "speakers"
                    / str(speaker_id)
                    / audio_path
                )
                audio_paths.append(str(full_audio_path))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def cb(index: int, progress: int):
                logger = get_logger()
                logger.query(
                    "UPDATE cleaning_run SET gen_file_embeddings_progress=? WHERE ID=?",
                    [progress, cleaning_run_id],
                )

            print("Generating file embeddings ... \n")

            gen_file_emeddings(
                in_paths=audio_paths,
                out_dir=str(Path(data_path) / "file_embeds"),
                callback=cb,
                device=device,
            )
            cur.execute(
                "UPDATE cleaning_run SET stage='detect_outliers', gen_file_embeddings_progress=1.0 WHERE ID=?",
                (cleaning_run_id,),
            )
            con.commit()

        elif stage == "detect_outliers":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            speaker2info = {}
            for (audio_path, speaker_id, text, sample_id) in cur.execute(
                """
                SELECT sample.audio_path, speaker.ID AS speakerID, sample.text AS text,
                sample.ID AS sampleID
                FROM sample
                INNER JOIN speaker ON sample.speaker_id = speaker.ID
                INNER JOIN dataset ON speaker.dataset_id = dataset.ID
                INNER JOIN cleaning_run ON cleaning_run.dataset_id = dataset.ID
                WHERE cleaning_run.ID=?
                """,
                (cleaning_run_id,),
            ).fetchall():
                file_embedding_path = (
                    Path(data_path)
                    / "file_embeds"
                    / str(speaker_id)
                    / f"{Path(audio_path).stem}.pt"
                )
                info = {
                    "path": file_embedding_path,
                    "text": text,
                    "sample_id": sample_id,
                }
                if speaker_id in speaker2info:
                    speaker2info[speaker_id].append(info)
                else:
                    speaker2info[speaker_id] = [info]

            infos, ys = [], []

            for i, key in enumerate(speaker2info):
                for info in speaker2info[key]:
                    infos.append(info)
                    ys.append(i)

            y = np.array(ys)
            n_classes = len(speaker2info.keys())

            print("Loading Embeddings ...\n")
            embeddings = load_embeddings([info["path"] for info in infos])
            x = torch.stack(embeddings).numpy()

            print("\nSearching for noisy samples ...\n")
            indices, label_qualities = get_issues(x=x, y=y, n_classes=n_classes)
            for index, label_quality in zip(indices, label_qualities):
                print(index, label_quality)
                info = infos[index]
                cur.execute(
                    """INSERT INTO noisy_sample (label_quality, sample_id, cleaning_run_id) VALUES (?, ?, ?)""",
                    (label_quality, info["sample_id"], cleaning_run_id),
                )

            cur.execute(
                "UPDATE cleaning_run SET stage='choose_samples' WHERE ID=?",
                (cleaning_run_id,),
            )
            con.commit()

        elif stage == "choose_samples":
            pass

        elif stage == "apply_changes":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            cur.execute(
                "UPDATE cleaning_run SET stage='finished' WHERE ID=?",
                (cleaning_run_id,),
            )
            con.commit()

        elif stage == "finished":
            pass

        else:
            raise Exception(f"Stage stage '{stage}' is not a valid stage ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaning_run_id", type=int)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--preprocessing_runs_path", type=str)
    parser.add_argument("--datasets_path", type=str)
    args = parser.parse_args()

    continue_cleaning_run(
        cleaning_run_id=args.cleaning_run_id, 
        db_path=args.db_path, 
        preprocessing_runs_path=args.preprocessing_runs_path, 
        datasets_path=args.datasets_path
    )