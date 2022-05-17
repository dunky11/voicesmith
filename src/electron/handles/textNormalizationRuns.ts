import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  DATASET_DIR,
  DB_PATH,
  PREPROCESSING_RUNS_DIR,
  USER_DATA_PATH,
} from "../utils/globals";
import { db, getSpeakersWithSamples } from "../utils/db";
import {
  TextNormalizationRunConfigInterface,
  TextNormalizationInterface,
  SpeakerInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";

ipcMain.on(
  "continue-text-normalization-run",
  (event: IpcMainEvent, runID: number) => {
    startRun(event, "text_normalization_run.py", [
      "--text_normalization_run_id",
      String(runID),
      "--db_path",
      DB_PATH,
      "--preprocessing_runs_path",
      PREPROCESSING_RUNS_DIR,
      "--user_data_path",
      USER_DATA_PATH,
    ]);
  }
);

ipcMain.handle(
  "fetch-text-normalization-run",
  (event: IpcMainInvokeEvent, ID: number) => {
    const run: TextNormalizationInterface = db
      .prepare(
        "SELECT ID, name, stage, language FROM text_normalization_run WHERE ID=@ID"
      )
      .get({ ID });
    return run;
  }
);

ipcMain.handle(
  "update-text-normalization-run-config",
  (
    event: IpcMainInvokeEvent,
    ID: number,
    config: TextNormalizationRunConfigInterface
  ) => {
    return db
      .prepare(
        "UPDATE text_normalization_run SET name=@name, language=@language, dataset_id=@datasetID WHERE ID=@ID"
      )
      .run({
        ID,
        ...config,
      });
  }
);

ipcMain.handle(
  "fetch-text-normalization-run-config",
  (event: IpcMainInvokeEvent, ID: number) => {
    return db
      .prepare(
        "SELECT name, dataset_id AS datasetID, language FROM text_normalization_run WHERE ID=@ID"
      )
      .get({ ID });
  }
);

ipcMain.handle(
  "fetch-text-normalization-samples",
  (event: IpcMainInvokeEvent, ID: number) => {
    return db
      .prepare(
        `
      SELECT text_normalization_sample.ID AS ID, text_normalization_sample.old_text AS oldText,
      text_normalization_sample.new_text AS newText, sample.audio_path AS audioPath, speaker.ID AS speakerID, 
      dataset.ID AS datasetID, text_normalization_sample.reason
      FROM text_normalization_sample 
      INNER JOIN sample ON text_normalization_sample.sample_id = sample.ID
      INNER JOIN speaker ON sample.speaker_id = speaker.ID
      INNER JOIN dataset ON speaker.dataset_id = dataset.ID
      WHERE text_normalization_sample.text_normalization_run_id = @ID
      `
      )
      .all({ ID })
      .map((el: any) => ({
        ID: el.ID,
        oldText: el.oldText,
        newText: el.newText,
        reason: el.reason,
        audioPath: path.join(
          DATASET_DIR,
          String(el.datasetID),
          "speakers",
          String(el.speakerID),
          el.audioPath
        ),
      }));
  }
);

ipcMain.handle(
  "remove-text-normalization-samples",
  (event: IpcMainInvokeEvent, sampleIDs: number[]) => {
    const removeSample = db.prepare(
      "DELETE FROM text_normalization_sample WHERE ID=@sampleID"
    );
    db.transaction(() => {
      for (const sampleID of sampleIDs) {
        removeSample.run({ sampleID });
      }
    })();
  }
);

ipcMain.on(
  "finish-text-normalization-run",
  (event: IpcMainEvent, runID: number) => {
    const samples = db
      .prepare(
        "SELECT new_text AS newText, sample_id AS sampleID FROM text_normalization_sample WHERE text_normalization_run_id=@runID"
      )
      .all({ runID });
    const updateSampleStmt = db.prepare(
      "UPDATE sample SET text=@text WHERE ID=@ID"
    );
    db.transaction(() => {
      for (const sample of samples) {
        updateSampleStmt.run({ text: sample.newText, ID: sample.sampleID });
      }
    })();
    event.reply("finish-text-normalization-run-reply", {
      type: "finished",
    });
  }
);
