import { app, ipcMain, IpcMainEvent } from "electron";
import fsNative from "fs";
import os from "os";
const fsPromises = fsNative.promises;
import { SettingsInterface, AppInfoInterface } from "../../interfaces";
import { UserDataPath } from "../utils/globals";
import { DB } from "../utils/db";
import { copyDir } from "../utils/files";

ipcMain.handle("get-app-info", (event: IpcMainEvent) => {
  const info: AppInfoInterface = {
    version: app.getVersion(),
    platform: process.platform,
  };
  return info;
});

ipcMain.on(
  "save-settings",
  async (event: IpcMainEvent, settings: SettingsInterface) => {
    const from = UserDataPath().getPath();
    const updatePaths = from !== settings.dataPath;
    if (updatePaths) {
      await copyDir(from, settings.dataPath);
    }
    DB.getInstance()
      .prepare("UPDATE settings SET data_path=@dataPath WHERE ID=1")
      .run(settings);
    if (updatePaths) {
      await fsPromises.rmdir(from, { recursive: true });
      UserDataPath().setPath(settings.dataPath);
    }
    event.reply("save-settings-reply", { type: "finished" });
  }
);

ipcMain.handle("fetch-settings", () => {
  let settings = DB.getInstance()
    .prepare("SELECT data_path AS dataPath FROM settings WHERE ID=1")
    .get();
  settings = {
    ...settings,
    dataPath:
      settings.dataPath === null ? UserDataPath().getPath() : settings.dataPath,
  };
  return settings;
});
