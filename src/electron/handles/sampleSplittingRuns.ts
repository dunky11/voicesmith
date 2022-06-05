import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  FINISH_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
  FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL,
} from "../../channels";
import {
  getDatasetsDir,
  DB_PATH,
  UserDataPath,
  ASSETS_PATH,
  getSampleSplittingRunsDir,
  getModelsDir,
} from "../utils/globals";
import { DB } from "../utils/db";
import {
  SampleSplittingRunInterface,
  SampleSplittingSampleInterface,
} from "../../interfaces";
import { startRun } from "../utils/processes";
import { CONDA_ENV_NAME } from "../../config";

ipcMain.on(
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    startRun(
      event,
      "sample_splitting_run.py",
      [
        "--run_id",
        String(runID),
        "--db_path",
        DB_PATH,
        "--assets_path",
        ASSETS_PATH,
        "--preprocessing_runs_dir",
        getSampleSplittingRunsDir(),
        "--datasets_path",
        getDatasetsDir(),
        "--environment_name",
        CONDA_ENV_NAME,
      ],
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
        creating_splits_progress AS creatingSplitsProgress, dataset_id AS datasetID
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
        datasetID: run.datasetID,
        maximumWorkers: run.maximumWorkers,
        ID: run.ID,
        device: run.device,
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
  (event: IpcMainEvent, sampleIDs: number[]) => {
    console.log("HERE");
    console.log(sampleIDs);
    const rmSampleStmt = DB.getInstance().prepare(
      "DELETE FROM sample_splitting_run_split WHERE ID=@ID"
    );
    DB.getInstance().transaction(() => {
      for (const ID of sampleIDs) {
        rmSampleStmt.run({ ID });
      }
    })();
  }
);

ipcMain.handle(
  FINISH_SAMPLE_SPLITTING_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    // TODO
    /**
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
   */
  }
);
