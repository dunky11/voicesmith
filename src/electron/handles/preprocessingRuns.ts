import { ipcMain, IpcMainInvokeEvent } from "electron";
import path from "path";
import {
  CREATE_PREPROCESSING_RUN_CHANNEL,
  FETCH_PREPROCESSING_RUNS_CHANNEL,
  EDIT_PREPROCESSING_RUN_NAME_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
  FETCH_PREPROCESSING_NAMES_USED_CHANNEL,
  FETCH_PREPROCESSING_RUNS_CHANNEL_TYPES,
} from "../../channels";
import { fetchCleaningRuns } from "./cleaningRuns";
import { fetchSampleSplittingRuns } from "./sampleSplittingRuns";
import { fetchTextNormalizationRuns } from "./textNormalizationRuns";
import { safeRmDir } from "../utils/files";
import { RunInterface, PreprocessingRunType } from "../../interfaces";
import {
  getCleaningRunsDir,
  getSampleSplittingRunsDir,
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

ipcMain.handle(
  FETCH_PREPROCESSING_RUNS_CHANNEL.IN,
  (): FETCH_PREPROCESSING_RUNS_CHANNEL_TYPES["IN"]["OUT"] => {
    const cleaningRuns = fetchCleaningRuns();
    const textNormalizationRuns = fetchTextNormalizationRuns();
    const sampleSplittingRuns = fetchSampleSplittingRuns();
    const ret: PreprocessingRunType[] = [
      ...cleaningRuns,
      ...textNormalizationRuns,
      ...sampleSplittingRuns,
    ];
    return ret;
  }
);

ipcMain.handle(
  EDIT_PREPROCESSING_RUN_NAME_CHANNEL.IN,
  (
    event: IpcMainInvokeEvent,
    preprocessingRun: RunInterface,
    newName: string
  ) => {
    switch (preprocessingRun.type) {
      case "cleaningRun":
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
  async (event: IpcMainInvokeEvent, preprocessingRun: RunInterface) => {
    switch (preprocessingRun.type) {
      case "cleaningRun": {
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
        safeRmDir(path.join(getCleaningRunsDir(), String(preprocessingRun.ID)));
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
        safeRmDir(
          path.join(getTextNormalizationRunsDir(), String(preprocessingRun.ID))
        );
        break;
      }
      case "sampleSplittingRun":
        DB.getInstance().transaction(() => {
          DB.getInstance()
            .prepare(
              "DELETE FROM sample_splitting_run_sample WHERE sample_splitting_run_sample.sample_splitting_run_id=@ID"
            )
            .run({
              ID: preprocessingRun.ID,
            });
          DB.getInstance()
            .prepare("DELETE FROM sample_splitting_run WHERE ID=@ID")
            .run({
              ID: preprocessingRun.ID,
            });
        })();
        safeRmDir(
          path.join(getSampleSplittingRunsDir(), String(preprocessingRun.ID))
        );
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
    for (const tableName of [
      "cleaning_run",
      "text_normalization_run",
      "sample_splitting_run",
    ]) {
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
