import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  FINISH_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
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
import { SampleSplittingRunInterface } from "../../interfaces";
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
