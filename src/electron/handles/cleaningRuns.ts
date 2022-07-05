import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  CONTINUE_CLEANING_RUN_CHANNEL,
  FETCH_CLEANING_RUNS_CHANNEL,
  FETCH_CLEANING_RUNS_CHANNEL_TYPES,
  UPDATE_CLEANING_RUN_CONFIG_CHANNEL,
  FETCH_CLEANING_RUN_SAMPLES_CHANNEL,
  FINISH_CLEANING_RUN_CHANNEL,
  FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES,
  REMOVE_CLEANING_RUN_SAMPLES_CHANNEL,
  REMOVE_CLEANING_RUN_SAMPLES_CHANNEL_TYPES,
} from "../../channels";
import { safeUnlink } from "../utils/files";
import { getDatasetsDir } from "../utils/globals";
import { bool2int, DB, getSpeakersWithSamples } from "../utils/db";
import {
  CleaningRunInterface,
  CleaningRunConfigInterface,
  SpeakerInterface,
  ContinueTrainingRunReplyInterface,
  FinishCleaningRunReplyInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";

ipcMain.on(
  CONTINUE_CLEANING_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    const datasetID = DB.getInstance()
      .prepare(
        "SELECT dataset_id AS datasetID FROM cleaning_run WHERE ID=@runID"
      )
      .get({ runID }).datasetID;
    const speakers = getSpeakersWithSamples(datasetID);

    let sampleCount = 0;
    speakers.forEach((speaker: SpeakerInterface) => {
      sampleCount += speaker.samples.length;
    });

    if (sampleCount == 0) {
      const reply: ContinueTrainingRunReplyInterface = {
        type: "notEnoughSamples",
      };
      event.reply(CONTINUE_CLEANING_RUN_CHANNEL.REPLY, reply);
      return;
    }
    startRun(
      event,
      "./backend/voice_smith/cleaning_run.py",
      ["--run_id", String(runID)],
      false
    );
  }
);

export const fetchCleaningRuns = (
  ID: number | null = null
): CleaningRunInterface[] => {
  const fetchStmt = DB.getInstance().prepare(
    `SELECT cleaning_run.ID AS ID, cleaning_run.name AS name, stage,
      dataset.ID AS datasetID, dataset.name AS datasetName,
      cleaning_run.skip_on_error AS skipOnError,
      copying_files_progress AS copyingFilesProgress,
      transcription_progress AS transcriptionProgress,
      applying_changes_progress AS applyingChangesProgress,
      maximum_workers AS maximumWorkers,
      device
    FROM cleaning_run
    LEFT JOIN dataset ON cleaning_run.dataset_id = dataset.ID
    ${ID === null ? "" : "WHERE cleaning_run.ID=@ID"}`
  );
  let runsRaw;
  if (ID === null) {
    runsRaw = fetchStmt.all();
  } else {
    runsRaw = [fetchStmt.get({ ID })];
  }
  return runsRaw.map((el: any) => {
    const run: CleaningRunInterface = {
      ID: el.ID,
      stage: el.stage,
      type: "cleaningRun",
      name: el.name,
      configuration: {
        name: el.name,
        datasetID: el.datasetID,
        datasetName: el.datasetName,
        skipOnError: el.skipOnError === 1,
        device: el.device,
        maximumWorkers: el.maximumWorkers,
      },
      copyingFilesProgress: el.copyingFilesProgress,
      transcriptionProgress: el.transcriptionProgress,
      applyingChangesProgress: el.applyingChangesProgress,
      canStart: el.datasetID !== null,
    };
    return run;
  });
};

ipcMain.handle(
  FETCH_CLEANING_RUNS_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    { ID }: FETCH_CLEANING_RUNS_CHANNEL_TYPES["IN"]["ARGS"]
  ): FETCH_CLEANING_RUNS_CHANNEL_TYPES["IN"]["OUT"] => {
    return fetchCleaningRuns(ID);
  }
);

ipcMain.handle(
  UPDATE_CLEANING_RUN_CONFIG_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    ID: number,
    config: CleaningRunConfigInterface
  ) => {
    return DB.getInstance()
      .prepare(
        `
        UPDATE cleaning_run 
          SET name=@name, 
          dataset_id=@datasetID,
          skip_on_error=@skipOnError,
          device=@device,
          maximum_workers=@maximumWorkers
        WHERE ID=@ID
        `
      )
      .run(
        bool2int({
          ID,
          ...config,
        })
      );
  }
);

ipcMain.handle(
  FETCH_CLEANING_RUN_SAMPLES_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    { runID }: FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["ARGS"]
  ): FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["OUT"] => {
    return DB.getInstance()
      .prepare(
        `
      SELECT cleaning_run_sample.ID AS ID, cleaning_run_sample.quality_score AS qualityScore, 
      sample.text, sample.audio_path AS audioPath, speaker.ID AS speakerID, dataset.ID AS datasetID 
      FROM cleaning_run_sample
      INNER JOIN sample ON cleaning_run_sample.sample_id = sample.ID
      INNER JOIN speaker ON sample.speaker_id = speaker.ID
      INNER JOIN dataset ON speaker.dataset_id = dataset.ID
      WHERE cleaning_run_sample.cleaning_run_id = @runID
      ORDER BY cleaning_run_sample.quality_score ASC
      `
      )
      .all({ runID })
      .map((el: any) => ({
        ID: el.ID,
        text: el.text,
        qualityScore: el.qualityScore,
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
  REMOVE_CLEANING_RUN_SAMPLES_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    { sampleIDs }: REMOVE_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["ARGS"]
  ) => {
    const removeSample = DB.getInstance().prepare(
      "DELETE FROM cleaning_run_sample WHERE ID=@sampleID"
    );
    DB.getInstance().transaction(() => {
      for (const sampleID of sampleIDs) {
        removeSample.run({ sampleID });
      }
    })();
  }
);

ipcMain.on(
  FINISH_CLEANING_RUN_CHANNEL.IN,
  async (event: IpcMainEvent, cleaningRunID: number) => {
    const samples = DB.getInstance()
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
    const deleteSampleStmt = DB.getInstance().prepare(
      "DELETE FROM sample WHERE ID=@sampleID"
    );
    DB.getInstance().transaction(() => {
      DB.getInstance()
        .prepare(
          "DELETE FROM noisy_sample WHERE cleaning_run_id=@cleaningRunID"
        )
        .run({ cleaningRunID });
      for (const sample of samples) {
        deleteSampleStmt.run({ sampleID: sample.sampleID });
      }
    })();
    const reply: FinishCleaningRunReplyInterface = {
      type: "progress",
      progress: 0.5,
    };
    event.reply(FINISH_CLEANING_RUN_CHANNEL.REPLY, reply);
    for (let i = 0; i < samples.length; i++) {
      const reply: FinishCleaningRunReplyInterface = {
        type: "progress",
        progress: 0.5 + i / samples.length / 2,
      };
      event.reply(FINISH_CLEANING_RUN_CHANNEL.REPLY, reply);
      const sample = samples[i];
      const audioPath = path.join(
        getDatasetsDir(),
        String(sample.datasetID),
        "speakers",
        String(sample.speakerID),
        sample.audioPath
      );
      await safeUnlink(audioPath);
    }
    const finishReply: FinishCleaningRunReplyInterface = {
      type: "finished",
    };
    event.reply(FINISH_CLEANING_RUN_CHANNEL.REPLY, finishReply);
  }
);
