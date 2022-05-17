import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import { safeUnlink } from "../utils/files";
import { DATASET_DIR, DB_PATH, PREPROCESSING_RUNS_DIR } from "../utils/globals";
import { db, getSpeakersWithSamples } from "../utils/db";
import {
  DSCleaningInterface,
  CleaningRunConfigInterface,
  SpeakerInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";

ipcMain.on("continue-cleaning-run", (event: IpcMainEvent, runID: number) => {
  const datasetID = db
    .prepare("SELECT dataset_id AS datasetID FROM cleaning_run WHERE ID=@runID")
    .get({ runID }).datasetID;
  const speakers = getSpeakersWithSamples(datasetID);

  if (speakers.length <= 1) {
    event.reply("continue-run-reply", {
      type: "notEnoughSpeakers",
    });
    return;
  }

  let sampleCount = 0;
  speakers.forEach((speaker: SpeakerInterface) => {
    sampleCount += speaker.samples.length;
  });

  if (sampleCount == 0) {
    event.reply("continue-run-reply", {
      type: "notEnoughSamples",
    });
    return;
  }
  startRun(event, "cleaning_run.py", [
    "--cleaning_run_id",
    String(runID),
    "--db_path",
    DB_PATH,
    "--preprocessing_runs_path",
    PREPROCESSING_RUNS_DIR,
    "--datasets_path",
    DATASET_DIR,
  ]);
});

ipcMain.handle(
  "fetch-cleaning-run",
  (event: IpcMainInvokeEvent, ID: number) => {
    const run: DSCleaningInterface = db
      .prepare("SELECT ID, name, stage FROM cleaning_run WHERE ID=@ID")
      .get({ ID });
    return run;
  }
);

ipcMain.handle(
  "fetch-cleaning-run-config",
  (event: IpcMainInvokeEvent, ID: number) => {
    return db
      .prepare(
        "SELECT name, dataset_id AS datasetID FROM cleaning_run WHERE ID=@ID"
      )
      .get({ ID });
  }
);

ipcMain.handle(
  "update-cleaning-run-config",
  (
    event: IpcMainInvokeEvent,
    ID: number,
    config: CleaningRunConfigInterface
  ) => {
    return db
      .prepare(
        "UPDATE cleaning_run SET name=@name, dataset_id=@datasetID WHERE ID=@ID"
      )
      .run({
        ID,
        ...config,
      });
  }
);

ipcMain.handle(
  "fetch-noisy-samples",
  (event: IpcMainInvokeEvent, cleaningRunID: number) => {
    return db
      .prepare(
        `
      SELECT noisy_sample.ID AS ID, noisy_sample.label_quality AS labelQuality, 
      sample.text, sample.audio_path AS audioPath, speaker.ID AS speakerID, dataset.ID AS datasetID 
      FROM noisy_sample
      INNER JOIN sample ON noisy_sample.sample_id = sample.ID
      INNER JOIN speaker ON sample.speaker_id = speaker.ID
      INNER JOIN dataset ON speaker.dataset_id = dataset.ID
      WHERE noisy_sample.cleaning_run_id = @cleaningRunID
      ORDER BY noisy_sample.label_quality ASC
      `
      )
      .all({ cleaningRunID })
      .map((el: any) => ({
        ID: el.ID,
        text: el.text,
        labelQuality: el.labelQuality,
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
  "remove-noisy-samples",
  (event: IpcMainInvokeEvent, sampleIDs: number[]) => {
    const removeSample = db.prepare(
      "DELETE FROM noisy_sample WHERE ID=@sampleID"
    );
    db.transaction(() => {
      for (const sampleID of sampleIDs) {
        removeSample.run({ sampleID });
      }
    })();
  }
);

ipcMain.on(
  "finish-cleaning-run",
  async (event: IpcMainEvent, cleaningRunID: number) => {
    const samples = db
      .prepare(
        `
      SELECT sample.ID AS sampleID, dataset.ID AS datasetID, speaker.ID AS speakerID, 
      sample.audio_path AS audioPath, sample.txt_path AS textPath FROM sample
      INNER JOIN noisy_sample ON noisy_sample.sample_id = sample.ID
      INNER JOIN speaker ON sample.speaker_id = speaker.ID
      INNER JOIN dataset ON speaker.dataset_id = dataset.ID
      WHERE noisy_sample.cleaning_run_id=@cleaningRunID
        `
      )
      .all({ cleaningRunID });
    const deleteSampleStmt = db.prepare(
      "DELETE FROM sample WHERE ID=@sampleID"
    );
    db.transaction(() => {
      db.prepare(
        "DELETE FROM noisy_sample WHERE cleaning_run_id=@cleaningRunID"
      ).run({ cleaningRunID });
      for (const sample of samples) {
        deleteSampleStmt.run({ sampleID: sample.sampleID });
      }
    })();
    event.reply("finish-cleaning-run-reply", {
      type: "progress",
      progress: 0.5,
    });
    for (let i = 0; i < samples.length; i++) {
      event.reply("finish-cleaning-run-reply", {
        type: "progress",
        progress: 0.5 + i / samples.length / 2,
      });
      const sample = samples[i];
      const audioPath = path.join(
        DATASET_DIR,
        String(sample.datasetID),
        "speakers",
        String(sample.speakerID),
        sample.audioPath
      );
      await safeUnlink(audioPath);
    }
    event.reply("finish-cleaning-run-reply", {
      type: "finished",
    });
  }
);
