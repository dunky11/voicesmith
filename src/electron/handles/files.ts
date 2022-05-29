import {
  ipcMain,
  IpcMainInvokeEvent,
  SaveDialogOptions,
  OpenDialogOptions,
  dialog,
} from "electron";
import path from "path";
import fsNative from "fs";
import fs from "fs-extra";
const fsPromises = fsNative.promises;
import {
  GET_IMAGE_DATA_URL_CHANNEL,
  GET_AUDIO_DATA_URL_CHANNEL,
  FETCH_LOGFILE_CHANNEL,
  EXPORT_FILES_CHANNEL,
  PICK_SINGLE_FOLDER_CHANNEL,
  EXPORT_FILE_CHANNEL,
} from "../../channels";
import {
  getTrainingRunsDir,
  getModelsDir,
  getCleaningRunsDir,
  getTextNormalizationRunsDir,
} from "../utils/globals";
import { DB } from "../utils/db";

ipcMain.handle(
  GET_IMAGE_DATA_URL_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, path: string) => {
    const extension = path.split(".").pop();
    let base64 = fs.readFileSync(path).toString("base64");
    base64 = `data:image/${extension};base64,${base64}`;
    return base64;
  }
);

ipcMain.handle(
  GET_AUDIO_DATA_URL_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, path: string) => {
    const extension = path.split(".").pop();
    let base64 = fs.readFileSync(path).toString("base64");
    base64 = `data:audio/${extension};base64,${base64}`;
    return base64;
  }
);

ipcMain.handle(
  FETCH_LOGFILE_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    name: string,
    fileName: string,
    type: string
  ) => {
    let dir;
    switch (type) {
      case "trainingRun":
        dir = getTrainingRunsDir();
        break;
      case "model":
        dir = getModelsDir();
        break;
      case "cleaningRun":
        dir = getCleaningRunsDir();
        break;
      case "textNormalizationRun":
        dir = getTextNormalizationRunsDir();
        break;
      default:
        throw new Error(
          `No case selected in switch-statement, '${type}' is not a valid case ... `
        );
    }
    const filePath = path.join(dir, name, "logs", fileName);
    const lines = await new Promise((resolve, reject) => {
      fs.readFile(filePath, "utf8", (err, data) => {
        if (err) {
          console.log(err);
          resolve([]);
          return;
        }
        const lines = data.split(/\r?\n/);
        resolve(lines);
      });
    });
    return lines;
  }
);

ipcMain.handle(
  EXPORT_FILES_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, inPaths: string[]) => {
    const options: OpenDialogOptions = {
      title: "Export Files",
      defaultPath: inPaths[0],
      buttonLabel: "Export",
      properties: ["openDirectory", "createDirectory"],
    };

    dialog.showOpenDialog(null, options).then(async ({ filePaths }) => {
      for (const inPath of inPaths) {
        await fsPromises.copyFile(
          inPath,
          path.join(filePaths[0], path.basename(inPath))
        );
      }
    });
  }
);

ipcMain.handle(
  PICK_SINGLE_FOLDER_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, inPath: string) => {
    const options: OpenDialogOptions = {
      title: "Set a new storage path",
      properties: ["openDirectory", "createDirectory"],
    };
    const filePath = await new Promise((resolve, reject) => {
      dialog.showOpenDialog(null, options).then(async ({ filePaths }) => {
        resolve(filePaths[0]);
      });
    });
    return filePath;
  }
);

ipcMain.handle(
  EXPORT_FILE_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, inPath: string) => {
    const options: SaveDialogOptions = {
      title: "Export File",
      defaultPath: inPath,
      buttonLabel: "Export",
      properties: ["createDirectory"],
    };

    dialog.showSaveDialog(null, options).then(async ({ filePath }) => {
      await fsPromises.copyFile(inPath, filePath);
    });
  }
);
