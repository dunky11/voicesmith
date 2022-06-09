import { ipcMain } from "electron";
import { START_BACKEND_CHANNEL } from "../../channels";
import { startContainer } from "../utils/docker";

ipcMain.handle(START_BACKEND_CHANNEL.IN, async () => {
  await startContainer(null, null);
});
