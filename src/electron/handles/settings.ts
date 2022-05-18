import { app, ipcMain } from "electron";
import { DB } from "../utils/db";

ipcMain.handle("fetch-settings", () => {
  let settings = DB.getInstance()
    .prepare("SELECT data_path AS dataPath FROM settings WHERE ID=1")
    .get();
  settings = {
    ...settings,
    dataPath: settings.dataPath === null ? app.getPath("userData") : null,
  };
  return settings;
});
