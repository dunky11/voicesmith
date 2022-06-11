import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  EDIT_TEXT_NORMALIZATION_SAMPLE_NEW_TEXT_CHANNEL,
  CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL,
  FETCH_TEXT_NORMALIZATION_RUN_CHANNEL,
  UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
  FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
  FETCH_TEXT_NORMALIZATION_SAMPLES_CHANNEL,
  FINISH_TEXT_NORMALIZATION_RUN_CHANNEL,
  REMOVE_TEXT_NORMALIZATION_SAMPLES_CHANNEL,
} from "../../channels";
import { getDatasetsDir } from "../utils/globals";
import { DB } from "../utils/db";
import {
  TextNormalizationRunConfigInterface,
  TextNormalizationInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";

ipcMain.on(
  CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    startRun(
      event,
      "/home/backend/voice_smith/text_normalization_run.py",
      ["--run_id", String(runID)],
      false
    );
  }
);

// TODO merge all different edits into one API
ipcMain.handle(
  EDIT_TEXT_NORMALIZATION_SAMPLE_NEW_TEXT_CHANNEL.IN,
  (event: IpcMainEvent, ID: number, newText: string) => {
    DB.getInstance()
      .prepare(
        "UPDATE text_normalization_sample SET new_text=@newText WHERE ID=@ID"
      )
      .run({ ID, newText });
  }
);

ipcMain.handle(
  FETCH_TEXT_NORMALIZATION_RUN_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number) => {
    const run: TextNormalizationInterface = DB.getInstance()
      .prepare(
        "SELECT ID, name, stage, language, text_normalization_progress AS textNormalizationProgress FROM text_normalization_run WHERE ID=@ID"
      )
      .get({ ID });
    return run;
  }
);

ipcMain.handle(
  UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    ID: number,
    config: TextNormalizationRunConfigInterface
  ) => {
    return DB.getInstance()
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
  FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number) => {
    return DB.getInstance()
      .prepare(
        "SELECT name, dataset_id AS datasetID, language FROM text_normalization_run WHERE ID=@ID"
      )
      .get({ ID });
  }
);

ipcMain.handle(
  FETCH_TEXT_NORMALIZATION_SAMPLES_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number) => {
    return DB.getInstance()
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
          getDatasetsDir(),
          String(el.datasetID),
          "speakers",
          String(el.speakerID),
          el.audioPath
        ),
      }));
  }
);

ipcMain.handle(
  REMOVE_TEXT_NORMALIZATION_SAMPLES_CHANNEL.IN,
  (event: IpcMainInvokeEvent, sampleIDs: number[]) => {
    const removeSample = DB.getInstance().prepare(
      "DELETE FROM text_normalization_sample WHERE ID=@sampleID"
    );
    DB.getInstance().transaction(() => {
      for (const sampleID of sampleIDs) {
        removeSample.run({ sampleID });
      }
    })();
  }
);

ipcMain.handle(
  FINISH_TEXT_NORMALIZATION_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    const samples = DB.getInstance()
      .prepare(
        "SELECT new_text AS newText, sample_id AS sampleID FROM text_normalization_sample WHERE text_normalization_run_id=@runID"
      )
      .all({ runID });
    const updateSampleStmt = DB.getInstance().prepare(
      "UPDATE sample SET text=@text WHERE ID=@ID"
    );
    DB.getInstance().transaction(() => {
      for (const sample of samples) {
        updateSampleStmt.run({ text: sample.newText, ID: sample.sampleID });
      }
    })();
  }
);
