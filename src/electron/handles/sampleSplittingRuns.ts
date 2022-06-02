import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import {
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_SAMPLE_CHANNEL,
  FINISH_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_SAMPLES_SPLITTING_RUNS_CHANNEL,
} from "../../channels";
import {
  getDatasetsDir,
  DB_PATH,
  getTextNormalizationRunsDir,
  UserDataPath,
  ASSETS_PATH,
} from "../utils/globals";
import { DB } from "../utils/db";
import { SampleSplittingRunInterface } from "../../interfaces";
import { startRun } from "../utils/processes";

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
        "--text_normalization_runs_path",
        getTextNormalizationRunsDir(),
        "--user_data_path",
        UserDataPath().getPath(),
        "--assets_path",
        ASSETS_PATH,
      ],
      false
    );
  }
);

export const FETCH_SAMPLE_SPLITTING_RUNS = (
  ID: number | null = null
): SampleSplittingRunInterface[] => {
  const query = `
      SELECT ID, name, stage, copying_files_progress AS copyingFilesProgress, 
        gen_vocab_progress AS genVocabProgress,  gen_align_progress AS genAlignProgress,
        splitting_samples_progress AS splittingSamplesProgress, dataset_id AS datasetID
      FROM sample_splittin_run ${ID === null ? "" : "WHERE ID=@ID"}`;
  if (ID === null) {
    return DB.getInstance().prepare(query).all();
  }
  return DB.getInstance().prepare(query).get({ ID });
};

ipcMain.handle(
  FETCH_SAMPLES_SPLITTING_RUNS_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number | null = null) => {
    return FETCH_SAMPLE_SPLITTING_RUNS(ID);
  }
);

ipcMain.handle(
  UPDATE_SAMPLE_SPLITTING_SAMPLE_CHANNEL.IN,
  (event: IpcMainInvokeEvent, run: SampleSplittingRunInterface) => {
    return DB.getInstance()
      .prepare(
        "UPDATE sample_splitting_run SET name=@name, dataset_id=@datasetID, maximum_workers=@maximumWorkers WHERE ID=@ID"
      )
      .run(run);
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
