import { ipcMain, IpcMainInvokeEvent } from "electron";
import path from "path";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import { exists } from "../utils/files";
import { PreprocessingRunInterface } from "../../interfaces";
import {
  CLEANING_RUNS_DIR,
  TEXT_NORMALIZATION_RUNS_DIR,
} from "../utils/globals";
import { db } from "../utils/db";

ipcMain.handle(
  "create-preprocessing-run",
  (event: IpcMainInvokeEvent, name: string, type: string) => {
    switch (type) {
      case "dSCleaning":
        db.prepare("INSERT INTO cleaning_run (name) VALUES (@name)").run({
          name,
        });
        break;
      case "textNormalization":
        db.prepare(
          "INSERT INTO text_normalization_run (name) VALUES (@name)"
        ).run({
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

ipcMain.handle("fetch-preprocessing-runs", (event: IpcMainInvokeEvent) => {
  const cleaningRuns = db
    .prepare(
      `SELECT cleaning_run.ID AS ID, cleaning_run.name AS name, stage, dataset_id, dataset.name AS datasetName FROM cleaning_run LEFT JOIN dataset ON cleaning_run.dataset_id = dataset.ID`
    )
    .all()
    .map((el) => ({
      ...el,
      type: "dSCleaning",
    }));
  const textNormalizationRuns = db
    .prepare(
      `SELECT text_normalization_run.ID AS ID, text_normalization_run.name AS name, stage, dataset_id FROM text_normalization_run LEFT JOIN dataset ON text_normalization_run.dataset_id = dataset.ID`
    )
    .all()
    .map((el) => ({
      ...el,
      type: "textNormalization",
    }));
  return cleaningRuns.concat(textNormalizationRuns);
});

ipcMain.handle(
  "edit-preprocessing-run-name",
  (
    event: IpcMainInvokeEvent,
    preprocessingRun: PreprocessingRunInterface,
    newName: string
  ) => {
    switch (preprocessingRun.type) {
      case "dSCleaning":
        db.prepare("UPDATE cleaning_run SET name=@name WHERE ID=@ID").run({
          ID: preprocessingRun.ID,
          name: newName,
        });
        break;
      case "textNormalization":
        db.prepare(
          "UPDATE text_normalization_run SET name=@name WHERE ID=@ID"
        ).run({
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
  "remove-preprocessing-run",
  async (
    event: IpcMainInvokeEvent,
    preprocessingRun: PreprocessingRunInterface
  ) => {
    switch (preprocessingRun.type) {
      case "dSCleaning": {
        db.transaction(() => {
          db.prepare("DELETE FROM noisy_sample WHERE cleaning_run_id=@ID").run({
            ID: preprocessingRun.ID,
          });
          db.prepare("DELETE FROM cleaning_run WHERE ID=@ID").run({
            ID: preprocessingRun.ID,
          });
        })();
        const dir = path.join(CLEANING_RUNS_DIR, String(preprocessingRun.ID));
        if (await exists(dir)) {
          await fsPromises.rmdir(dir, { recursive: true });
        }
        break;
      }
      case "textNormalization": {
        db.transaction(() => {
          db.prepare(
            "DELETE FROM text_normalization_sample WHERE text_normalization_run_id=@ID"
          ).run({
            ID: preprocessingRun.ID,
          });
          db.prepare("DELETE FROM text_normalization_run WHERE ID=@ID").run({
            ID: preprocessingRun.ID,
          });
        })();
        const dir = path.join(
          TEXT_NORMALIZATION_RUNS_DIR,
          String(preprocessingRun.ID)
        );
        if (await exists(dir)) {
          await fsPromises.rmdir(dir, { recursive: true });
        }
        break;
      }
      default: {
        throw new Error(
          `No case selected in switch-statement, '${preprocessingRun.type}' is not a valid case ...`
        );
      }
    }
  }
);

ipcMain.handle(
  "fetch-preprocessing-names-used",
  (event: IpcMainInvokeEvent, ID: number | null) => {
    const names: string[] = [];
    for (const tableName of ["cleaning_run", "text_normalization_run"]) {
      if (ID !== null) {
        db.prepare(`SELECT name FROM ${tableName} WHERE ID!=@ID`)
          .all({ ID })
          .forEach((el) => {
            names.push(el.name);
          });
      } else {
        db.prepare(`SELECT name FROM ${tableName}`)
          .all()
          .forEach((el) => {
            names.push(el.name);
          });
      }
    }
    return names;
  }
);
