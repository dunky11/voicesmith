import Database from "better-sqlite3";
import path from "path";
import { getDatasetsDir, DB_PATH } from "../utils/globals";
import { SpeakerSampleInterface } from "../../interfaces";
import { safeMkdir } from "./files";

const createTables = (db: any) => {
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS training_run (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        stage TEXT NOT NULL DEFAULT "not_started",
        maximum_workers INTEGER NOT NULL,
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
        device TEXT NOT NULL DEFAULT "CPU", 
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
    `
  ).run();
  db.prepare(
    `
      CREATE TABLE IF NOT EXISTS dataset (
          ID INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          UNIQUE(name)
      );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS speaker (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        dataset_id INTEGER NOT NULL,
        UNIQUE(name, dataset_id),
        FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
    ); 
    `
  ).run();
  db.prepare(
    `
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
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS image_statistic (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        step INTEGER NOT NULL,
        stage TEXT NOT NULL,
        training_run_id INTEGER NOT NULL,
        FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS audio_statistic (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        step INTEGER NOT NULL,
        stage TEXT NOT NULL,
        training_run_id INTEGER NOT NULL,
        FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS graph_statistic (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        step INTEGER NOT NULL,
        stage TEXT NOT NULL,
        value FLOAT NOT NULL,
        training_run_id INTEGER NOT NULL,
        FOREIGN KEY (training_run_id) REFERENCES training_run(ID)
    );
    `
  ).run();

  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS model (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(name)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS model_speaker (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        speaker_id INTEGER NOT NULL,
        model_id INTEGER NOT NULL,
        FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
        UNIQUE(model_id, name),
        UNIQUE(model_id, speaker_id)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS lexicon_word (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT NOT NULL,
        phonemes TEXT NOT NULL,
        model_id INTEGER NOT NULL,
        FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
        UNIQUE(model_id, word)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS symbol (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        symbol_id INTEGER NOT NULL,
        model_id INTEGER NOT NULL,
        FOREIGN KEY (model_id) REFERENCES model(ID) ON DELETE CASCADE,
        UNIQUE(model_id, symbol),
        UNIQUE(model_id, symbol_id)
    );
    `
  ).run();
  db.prepare(
    `
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
    `
  ).run();
  db.prepare(
    `
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
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS noisy_sample (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        label_quality FLOAT DEFAULT NULL,
        sample_id INT NOT NULL,
        cleaning_run_id INT NOT NULL,
        FOREIGN KEY (sample_id) REFERENCES sample(ID),
        FOREIGN KEY (cleaning_run_id) REFERENCES cleaning_run(ID)
    );
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS text_normalization_run (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        stage TEXT DEFAULT "not_started",
        language TEXT DEFAULT "en",
        text_normalization_progress FLOAT DEFAULT 0.0,
        dataset_id INTEGER DEFAULT NULL,  
        FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
    ); 
    `
  ).run();
  db.prepare(
    `
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
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS settings (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        data_path TEXT DEFAULT NULL,
        pid INTEGER DEFAULT NULL
    );
    `
  ).run();
  db.prepare(
    `
    INSERT OR IGNORE INTO settings (ID) VALUES (1) 
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS sample_splitting_run (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        maximum_workers INTEGER NOT NULL,
        name TEXT NOT NULL,
        stage TEXT DEFAULT "not_started",
        copying_files_progress FLOAT NOT NULL DEFAULT 0.0,
        gen_vocab_progress FLOAT NOT NULL DEFAULT 0.0,
        gen_align_progress FLOAT NOT NULL DEFAULT 0.0,
        creating_splits_progress FLOAT NOT NULL DEFAULT 0.0,
        device TEXT NOT NULL DEFAULT "CPU",
        dataset_id INTEGER DEFAULT NULL,
        FOREIGN KEY (dataset_id) REFERENCES dataset(ID)
    ); 
    `
  ).run();
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS sample_splitting_run_sample (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        sample_splitting_run_id INTEGER NOT NULL,
        sample_id INTEGER NOT NULL,
        FOREIGN KEY (sample_splitting_run_id) REFERENCES sample_splitting_run(ID),
        FOREIGN KEY (sample_id) REFERENCES sample(ID) 
    ); 
    `
  ).run();
  db.prepare(
    ` 
    CREATE TABLE IF NOT EXISTS sample_splitting_run_split (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        split_idx INTEGER NOT NULL,
        sample_splitting_run_sample_id INTEGER NOT NULL,
        FOREIGN KEY (sample_splitting_run_sample_id) REFERENCES sample_splitting_run_sample(ID) ON DELETE CASCADE
    ); 
    `
  ).run();
};

export const DB = (function () {
  let instance: any = null;
  return {
    getInstance: function () {
      if (instance == null) {
        safeMkdir(path.dirname(DB_PATH));
        instance = new Database(DB_PATH);
        createTables(instance);
        instance.constructor = null;
      }
      return instance;
    },
  };
})();

export const bool2int = (obj: { [key: string]: any }) => {
  for (const [key, value] of Object.entries(obj)) {
    if (value === true) {
      obj[key] = 1;
    } else if (value === false) {
      obj[key] = 0;
    }
  }
  return obj;
};

export const getSpeakersWithSamples = (datasetID: number) => {
  const samples = DB.getInstance()
    .prepare(
      `SELECT sample.text AS text, speaker.ID as speakerID, speaker.name, sample.txt_path AS txtPath, 
      sample.audio_path AS audioPath, sample.ID 
      FROM speaker 
      LEFT JOIN sample ON speaker.ID = sample.speaker_id
      WHERE speaker.dataset_id=@datasetID`
    )
    .all({ datasetID })
    .map((sample: any) => ({
      text: sample.text,
      speakerID: sample.speakerID,
      name: sample.name,
      txtPath: sample.txtPath,
      audioPath: sample.audioPath,
      ID: sample.ID,
      fullAudioPath:
        sample.txtPath === null
          ? null
          : path.join(
              getDatasetsDir(),
              String(datasetID),
              "speakers",
              String(sample.speakerID),
              sample.audioPath
            ),
    }));

  const speaker2Samples: { [key: string]: SpeakerSampleInterface[] } = {};
  const speaker2SpeakerID: { [key: string]: number } = {};
  samples.forEach((sample: any) => {
    const name = sample.name;
    const speakerID = sample.speakerID;
    delete sample.name;
    delete sample.speakerID;
    if (name in speaker2Samples) {
      if (sample.txtPath != null) {
        speaker2Samples[name].push(sample);
      }
    } else {
      if (sample.txtPath == null) {
        speaker2Samples[name] = [];
      } else {
        speaker2Samples[name] = [sample];
      }
      speaker2SpeakerID[name] = speakerID;
    }
  });
  const speakers = Object.keys(speaker2Samples).map((key) => ({
    ID: speaker2SpeakerID[key],
    name: key,
    samples: speaker2Samples[key],
  }));
  return speakers;
};

export const getReferencedBy = (datasetID: number) => {
  let row = DB.getInstance()
    .prepare("SELECT name FROM training_run WHERE dataset_id=@datasetID")
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  row = DB.getInstance()
    .prepare("SELECT name FROM cleaning_run WHERE dataset_id=@datasetID")
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  row = DB.getInstance()
    .prepare(
      "SELECT name FROM text_normalization_run WHERE dataset_id=@datasetID"
    )
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  return null;
};
