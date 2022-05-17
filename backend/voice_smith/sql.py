import sqlite3
import pathlib
from pathlib import Path

def create_tables(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL DEFAULT "not_started",
            name TEXT NOT NULL,
            validation_size FLOAT NOT NULL,
            min_seconds FLOAT NOT NULL,
            max_seconds FLOAT NOT NULL, 
            use_audio_normalization BOOLEAN NOT NULL,
            acoustic_learning_rate FLOAT NOT NULL,
            acoustic_training_iterations BIGINT NOT NULL,
            acoustic_batch_size INTEGER NOT NULL,
            acoustic_grad_accum_steps INTEGER NOT NULL,
            acoustic_validate_every INTEGER NOT NULL,
            device TEXT NOT NULL, 
            vocoder_learning_rate FLOAT NOT NULL,
            vocoder_training_iterations BIGINT NOT NULL,
            vocoder_batch_size INTEGER NOT NULL,
            vocoder_grad_accum_steps INTEGER NOT NULL,
            vocoder_validate_every INTEGER NOT NULL,
            preprocessing_stage TEXT DEFAULT "not_started" NOT NULL,
            preprocessing_copying_files_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_gen_vocab_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_gen_align_progress FLOAT NOT NULL DEFAULT 0.0,
            preprocessing_extract_data_progress FLOAT NOT NULL DEFAULT 0.0,
            acoustic_fine_tuning_progress FLOAT NOT NULL DEFAULT 0.0,
            ground_truth_alignment_progress FLOAT NOT NULL DEFAULT 0.0,
            vocoder_fine_tuning_progress FLOAT NOT NULL DEFAULT 0.0,
            save_model_progress FLOAT NOT NULL DEFAULT 0.0,
            only_train_speaker_emb_until INTEGER NOT NULL,
            dataset_id INTEGER DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID) ON DELETE SET NULL,
            UNIQUE(name)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            UNIQUE(name)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS speaker (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_id INTEGER NOT NULL,
            UNIQUE(name, dataset_id),
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        ); 
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sample (
            ID INTEGER PRIMARY KEY,
            txt_path TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            speaker_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            UNIQUE(txt_path, speaker_id),
            UNIQUE(audio_path, speaker_id), 
            FOREIGN KEY (speaker_id) REFERENCES speaker(ID)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS image_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS graph_statistic (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            step INTEGER NOT NULL,
            stage TEXT NOT NULL,
            value FLOAT NOT NULL,
            training_run_id INTEGER NOT NULL,
            FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_speaker (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            speaker_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, name),
            UNIQUE(model_id, speaker_id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS lexicon_word (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            phonemes TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, word)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS symbol (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            symbol_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
            UNIQUE(model_id, symbol),
            UNIQUE(model_id, symbol_id)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_synth (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            text TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            model_name TEXT NOT NULL,            
            created_at DEFAULT CURRENT_TIMESTAMP,
            sampling_rate INTEGER NOT NULL,
            dur_secs FLOAT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cleaning_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            get_txt_progress FLOAT DEFAULT 0.0,
            gen_file_embeddings_progress FLOAT DEFAULT 0.0,
            apply_changes_progress FLOAT DEFAULT 0.0,
            stage TEXT DEFAULT "not_started",
            dataset_id INTEGER DEFAULT NULL,
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS noisy_sample (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            label_quality FLOAT DEFAULT NULL,
            sample_id INT NOT NULL,
            cleaning_run_id INT NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES sample(ID),
            FOREIGN KEY (cleaning_run_id) REFERENCES cleaning_run(ID)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS text_normalization_run (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            stage TEXT DEFAULT "not_started",
            language TEXT DEFAULT "en",
            dataset_id INTEGER DEFAULT NULL,  
            FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
        ); 
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS text_normalization_sample (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            old_text TEXT NOT NULL,
            new_text TEXT NOT NULL,
            reason TEXT NOT NULL,
            sample_id INT NOT NULL,
            text_normalization_run_id INT NOT NULL,
            FOREIGN KEY (sample_id) REFERENCES sample(ID),
            FOREIGN KEY (text_normalization_run_id) REFERENCES text_normalization_run(ID)
        );
        """
    )


def get_con(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    return con
