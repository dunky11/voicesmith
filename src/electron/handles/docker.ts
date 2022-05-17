import { ipcMain } from "electron";
import childProcess from "child_process";

ipcMain.handle("fetch-has-docker", async () => {
  return await new Promise((resolve, reject) => {
    childProcess.exec("docker info", (error, stdout, stderr) => {
      return resolve(error === null);
    });
  });
});
