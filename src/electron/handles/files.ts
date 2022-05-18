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
  getAudioSynthDir,
  getTrainingRunsDir,
  getModelsDir,
  getCleaningRunsDir,
  getTextNormalizationRunsDir,
} from "../utils/globals";
import { DB } from "../utils/db";

ipcMain.handle(
  "get-image-data-url",
  async (event: IpcMainInvokeEvent, path: string) => {
    const extension = path.split(".").pop();
    let base64 = fs.readFileSync(path).toString("base64");
    base64 = `data:image/${extension};base64,${base64}`;
    return base64;
  }
);

ipcMain.handle(
  "get-audio-data-url",
  async (event: IpcMainInvokeEvent, path: string) => {
    const extension = path.split(".").pop();
    let base64 = fs.readFileSync(path).toString("base64");
    base64 = `data:audio/${extension};base64,${base64}`;
    return base64;
  }
);

ipcMain.handle("fetch-audios-synth", async (event: IpcMainInvokeEvent) => {
  const audios = DB.getInstance()
    .prepare(
      `
        SELECT 
        ID, 
        file_name AS fileName, 
        text, 
        speaker_name AS speakerName, 
        model_name AS modelName,
        created_at AS createdAt,
        sampling_rate as samplingRate,
        dur_secs AS durSecs
        FROM audio_synth
        ORDER BY created_at DESC
      `
    )
    .all()
    .map((audio: any) => {
      audio.filePath = path.join(getAudioSynthDir(), audio.fileName);
      delete audio.fileName;
      return audio;
    });
  return audios;
});

ipcMain.handle(
  "fetch-logfile",
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
  "export-files",
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
  "pick-single-folder",
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
  "export-file",
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
