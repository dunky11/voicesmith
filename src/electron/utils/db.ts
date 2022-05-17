import Database from "better-sqlite3";
import path from "path";
import { DATASET_DIR, DB_PATH } from "../utils/globals";
import { SpeakerSampleInterface } from "../../interfaces";

export const db = new Database(DB_PATH);

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
  const samples = db
    .prepare(
      `SELECT sample.text AS text, speaker.ID as speakerID, speaker.name, sample.txt_path AS txtPath, 
      sample.audio_path AS audioPath, sample.ID 
      FROM speaker 
      LEFT JOIN sample ON speaker.ID = sample.speaker_id
      WHERE speaker.dataset_id=@datasetID`
    )
    .all({ datasetID })
    .map((sample) => ({
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
              DATASET_DIR,
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
  let row = db
    .prepare("SELECT name FROM training_run WHERE dataset_id=@datasetID")
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  row = db
    .prepare("SELECT name FROM cleaning_run WHERE dataset_id=@datasetID")
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  row = db
    .prepare(
      "SELECT name FROM text_normalization_run WHERE dataset_id=@datasetID"
    )
    .get({ datasetID });
  if (row !== undefined) {
    return row.name;
  }
  return null;
};
