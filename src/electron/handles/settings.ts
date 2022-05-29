import { app, ipcMain, IpcMainEvent } from "electron";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import {
  GET_APP_INFO_CHANNEL,
  SAVE_SETTINGS_CHANNEL,
  FETCH_SETTINGS_CHANNEL,
} from "../../channels";
import { SettingsInterface, AppInfoInterface } from "../../interfaces";
import { UserDataPath } from "../utils/globals";
import { DB } from "../utils/db";
import { copyDir } from "../utils/files";

ipcMain.handle(GET_APP_INFO_CHANNEL.IN, (event: IpcMainEvent) => {
  const info: AppInfoInterface = {
    version: app.getVersion(),
    platform: process.platform,
  };
  return info;
});

ipcMain.on(
  SAVE_SETTINGS_CHANNEL.IN,
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
    event.reply(SAVE_SETTINGS_CHANNEL.REPLY, { type: "finished" });
  }
);

ipcMain.handle(FETCH_SETTINGS_CHANNEL.IN, () => {
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
