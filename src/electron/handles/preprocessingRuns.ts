import { ipcMain, IpcMainInvokeEvent } from "electron";
import path from "path";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import {
  CREATE_PREPROCESSING_RUN_CHANNEL,
  FETCH_PREPROCESSING_RUNS_CHANNEL,
  EDIT_PREPROCESSING_RUN_NAME_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
  FETCH_PREPROCESSING_NAMES_USED_CHANNEL,
} from "../../channels";
import { fetchSampleSplittingRuns } from "./sampleSplittingRuns";
import { exists } from "../utils/files";
import {
  PreprocessingRunInterface,
  SampleSplittingRunInterface,
} from "../../interfaces";
import {
  getCleaningRunsDir,
  getTextNormalizationRunsDir,
} from "../utils/globals";
import { DB } from "../utils/db";

ipcMain.handle(
  CREATE_PREPROCESSING_RUN_CHANNEL.IN,
  (event: IpcMainInvokeEvent, name: string, type: string) => {
    switch (type) {
      case "dSCleaningRun":
        DB.getInstance()
          .prepare("INSERT INTO cleaning_run (name) VALUES (@name)")
          .run({
            name,
          });
        break;
      case "textNormalizationRun":
        DB.getInstance()
          .prepare("INSERT INTO text_normalization_run (name) VALUES (@name)")
          .run({
            name,
          });
        break;
      case "sampleSplittingRun":
        DB.getInstance()
          .prepare(
            "INSERT INTO sample_splitting_run (name, maximum_workers) VALUES (@name, -1)"
          )
          .run({
            name,
          });
        break;
      default:
        throw new Error(
          `No case selected in switch-statement, '${type}' is not a valid case ...`
        );
    }
  }
);

ipcMain.handle(FETCH_PREPROCESSING_RUNS_CHANNEL.IN, () => {
  const cleaningRuns = DB.getInstance()
    .prepare(
      `SELECT cleaning_run.ID AS ID, cleaning_run.name AS name, stage, dataset_id, dataset.name AS datasetName FROM cleaning_run LEFT JOIN dataset ON cleaning_run.dataset_id = dataset.ID`
    )
    .all()
    .map((el: any) => ({
      ...el,
      type: "dSCleaningRun",
    }));
  const textNormalizationRuns = DB.getInstance()
    .prepare(
      `SELECT text_normalization_run.ID AS ID, text_normalization_run.name AS name, stage, dataset_id FROM text_normalization_run LEFT JOIN dataset ON text_normalization_run.dataset_id = dataset.ID`
    )
    .all()
    .map((el: any) => ({
      ...el,
      type: "textNormalizationRun",
    }));
  const sampleSplittingRuns = fetchSampleSplittingRuns().map(
    (el: SampleSplittingRunInterface) => ({ ...el, type: "sampleSplittingRun" })
  );
  console.log(sampleSplittingRuns);
  return cleaningRuns.concat(textNormalizationRuns).concat(sampleSplittingRuns);
});

ipcMain.handle(
  EDIT_PREPROCESSING_RUN_NAME_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    preprocessingRun: PreprocessingRunInterface,
    newName: string
  ) => {
    switch (preprocessingRun.type) {
      case "dSCleaningRun":
        DB.getInstance()
          .prepare("UPDATE cleaning_run SET name=@name WHERE ID=@ID")
          .run({
            ID: preprocessingRun.ID,
            name: newName,
          });
        break;
      case "textNormalizationRun":
        DB.getInstance()
          .prepare("UPDATE text_normalization_run SET name=@name WHERE ID=@ID")
          .run({
            ID: preprocessingRun.ID,
            name: newName,
          });
        break;
      case "sampleSplittingRun":
        DB.getInstance()
          .prepare("UPDATE sample_splitting_run SET name=@name WHERE ID=@ID")
          .run({
            ID: preprocessingRun.ID,
            name: newName,
          });
        break;
      default:
        throw new Error(
          `No case selected in switch-statement, '${preprocessingRun.type}' is not a valid case ...`
        );
    }
  }
);

ipcMain.handle(
  REMOVE_PREPROCESSING_RUN_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    preprocessingRun: PreprocessingRunInterface
  ) => {
    switch (preprocessingRun.type) {
      case "dSCleaningRun": {
        DB.getInstance().transaction(() => {
          DB.getInstance()
            .prepare("DELETE FROM noisy_sample WHERE cleaning_run_id=@ID")
            .run({
              ID: preprocessingRun.ID,
            });
          DB.getInstance()
            .prepare("DELETE FROM cleaning_run WHERE ID=@ID")
            .run({
              ID: preprocessingRun.ID,
            });
        })();
        const dir = path.join(
          getCleaningRunsDir(),
          String(preprocessingRun.ID)
        );
        if (await exists(dir)) {
          await fsPromises.rmdir(dir, { recursive: true });
        }
        break;
      }
      case "textNormalizationRun": {
        DB.getInstance().transaction(() => {
          DB.getInstance()
            .prepare(
              "DELETE FROM text_normalization_sample WHERE text_normalization_run_id=@ID"
            )
            .run({
              ID: preprocessingRun.ID,
            });
          DB.getInstance()
            .prepare("DELETE FROM text_normalization_run WHERE ID=@ID")
            .run({
              ID: preprocessingRun.ID,
            });
        })();
        const dir = path.join(
          getTextNormalizationRunsDir(),
          String(preprocessingRun.ID)
        );
        if (await exists(dir)) {
          await fsPromises.rmdir(dir, { recursive: true });
        }
        break;
      }
      case "sampleSplittingRun":
        DB.getInstance().transaction(() => {
          DB.getInstance()
            .prepare("DELETE FROM sample_splitting_run WHERE ID=@ID")
            .run({
              ID: preprocessingRun.ID,
            });
        })();
        break;
      default: {
        throw new Error(
          `No case selected in switch-statement, '${preprocessingRun.type}' is not a valid case ...`
        );
      }
    }
  }
);

ipcMain.handle(
  FETCH_PREPROCESSING_NAMES_USED_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number | null) => {
    const names: string[] = [];
    for (const tableName of ["cleaning_run", "text_normalization_run"]) {
      if (ID !== null) {
        DB.getInstance()
          .prepare(`SELECT name FROM ${tableName} WHERE ID!=@ID`)
          .all({ ID })
          .forEach((el: any) => {
            names.push(el.name);
          });
      } else {
        DB.getInstance()
          .prepare(`SELECT name FROM ${tableName}`)
          .all()
          .forEach((el: any) => {
            names.push(el.name);
          });
      }
    }
    return names;
  }
);
