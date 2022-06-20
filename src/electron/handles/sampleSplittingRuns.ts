import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_RUN_STAGE_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
  FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL,
} from "../../channels";
import { getDatasetsDir, getSampleSplittingRunsDir } from "../utils/globals";
import { DB } from "../utils/db";
import {
  SampleSplittingRunInterface,
  SampleSplittingSampleInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";

ipcMain.on(
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    startRun(
      event,
      "/home/backend/voice_smith/sample_splitting_run.py",
      ["--run_id", String(runID)],
      false
    );
  }
);

export const fetchSampleSplittingRuns = (
  ID: number | null = null
): SampleSplittingRunInterface[] => {
  const query = `
      SELECT sample_splitting_run.ID AS ID, device, sample_splitting_run.name AS name, 
        dataset.name AS datasetName, stage, maximum_workers AS maximumWorkers, 
        copying_files_progress AS copyingFilesProgress, gen_vocab_progress AS genVocabProgress,  
        gen_align_progress AS genAlignProgress,
        applying_changes_progress AS applyingChangesProgress,
        creating_splits_progress AS creatingSplitsProgress, 
        dataset_id AS datasetID
      FROM sample_splitting_run LEFT JOIN dataset ON dataset.ID = sample_splitting_run.dataset_id ${
        ID === null ? "" : "WHERE sample_splitting_run.ID=@ID"
      }`;
  if (ID === null) {
    return DB.getInstance().prepare(query).all();
  }
  return [DB.getInstance().prepare(query).get({ ID })];
};

ipcMain.handle(
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number | null = null) => {
    return fetchSampleSplittingRuns(ID);
  }
);

ipcMain.handle(
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL.IN,
  (event: IpcMainInvokeEvent, run: SampleSplittingRunInterface) => {
    DB.getInstance()
      .prepare(
        "UPDATE sample_splitting_run SET name=@name, dataset_id=@datasetID, maximum_workers=@maximumWorkers, device=@device WHERE ID=@ID"
      )
      .run({
        name: run.name,
        datasetID: run.configuration.datasetID,
        maximumWorkers: run.maximumWorkers,
        ID: run.ID,
        device: run.configuration.device,
      });
  }
);

ipcMain.handle(
  FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    const sample2Splits: { [id: number]: SampleSplittingSampleInterface } = {};
    DB.getInstance()
      .prepare(
        `
      SELECT sample.ID AS sampleID,
      sample_splitting_run_sample.ID AS sampleSplittingSampleID,
      sample_splitting_run_split.ID AS sampleSplittingRunSplitID,
      sample_splitting_run_sample.text AS sampleSplittingSampleText,
      sample_splitting_run_split.text AS sampleSplittingSplitText,
      sample_splitting_run_split.split_idx AS splitIdx,
      sample.audio_path AS audioPath,
      dataset.ID as datasetID,
      speaker.ID as speakerID,
      speaker.name AS speakerName
      FROM sample_splitting_run_split
      INNER JOIN sample_splitting_run_sample ON sample_splitting_run_split.sample_splitting_run_sample_id = sample_splitting_run_sample.ID
      INNER JOIN sample ON sample_splitting_run_sample.sample_id = sample.ID
      INNER JOIN speaker ON sample.speaker_id = speaker.ID
      INNER JOIN dataset ON  speaker.dataset_id = dataset.ID
      WHERE sample_splitting_run_sample.sample_splitting_run_id=@runID
      `
      )
      .all({ runID })
      .forEach((el: any) => {
        const split = {
          ID: el.sampleSplittingRunSplitID,
          text: el.sampleSplittingSplitText,
          audioPath: path.join(
            getSampleSplittingRunsDir(),
            String(runID),
            "splits",
            `${el.sampleSplittingSampleID}_split_${el.splitIdx}.flac`
          ),
        };
        if (sample2Splits[el.sampleSplittingSampleID] === undefined) {
          sample2Splits[el.sampleSplittingSampleID] = {
            ID: el.sampleSplittingSampleID,
            text: el.sampleSplittingSampleText,
            speakerName: el.speakerName,
            audioPath: path.join(
              getDatasetsDir(),
              String(el.datasetID),
              "speakers",
              String(el.speakerID),
              el.audioPath
            ),
            splits: [split],
          };
        } else {
          sample2Splits[el.sampleSplittingSampleID].splits.push(split);
        }
      });
    return Object.values(sample2Splits);
  }
);

ipcMain.handle(
  REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL.IN,
  (event: IpcMainEvent, sampleIDs: number[]) => {
    const rmSampleStmt = DB.getInstance().prepare(
      "DELETE FROM sample_splitting_run_sample WHERE ID=@ID"
    );
    DB.getInstance().transaction(() => {
      for (const ID of sampleIDs) {
        rmSampleStmt.run({ ID });
      }
    })();
  }
);

ipcMain.handle(
  REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL.IN,
  (event: IpcMainEvent, sampleID: number, splitIDs: number[]) => {
    const rmSplitStmt = DB.getInstance().prepare(
      "DELETE FROM sample_splitting_run_split WHERE ID=@ID"
    );
    const rmSampleStmt = DB.getInstance().prepare(
      "DELETE FROM sample_splitting_run_sample WHERE ID=@ID"
    );
    const countSplitStmt = DB.getInstance().prepare(
      `
      SELECT count(*) FROM sample_splitting_run_split 
      WHERE sample_splitting_run_split.sample_splitting_run_sample_id=@ID
    `
    );
    DB.getInstance().transaction(() => {
      for (const ID of splitIDs) {
        rmSplitStmt.run({ ID });
      }
      const count = countSplitStmt.get({ ID: sampleID })["count(*)"];
      if (count === 0) {
        rmSampleStmt.run({ ID: sampleID });
      }
    })();
  }
);

ipcMain.handle(
  UPDATE_SAMPLE_SPLITTING_RUN_STAGE_CHANNEL.IN,
  (
    event: IpcMainEvent,
    runID: number,
    stage: SampleSplittingRunInterface["stage"]
  ) => {
    DB.getInstance()
      .prepare("UPDATE sample_splitting_run SET stage=@stage WHERE ID=@runID")
      .run({ stage, runID });
  }
);
