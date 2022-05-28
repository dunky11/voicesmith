import { ipcMain, IpcMainEvent } from "electron";
import childProcess from "child_process";
import { getInstalledPath, CONDA_PATH } from "../utils/globals";
import {
  INSTALL_BACKEND_CHANNEL,
  FINISH_INSTALL_CHANNEL,
  FETCH_HAS_CONDA_CHANNEL,
  FETCH_NEEDS_INSTALL_CHANNEL,
} from "../../channels";
import { exists } from "../utils/files";
import { CONDA_ENV_NAME } from "../../config";
import fsNative from "fs";
import { InstallBackendReplyInterface } from "../../interfaces";
const fsPromises = fsNative.promises;

ipcMain.handle(FETCH_HAS_CONDA_CHANNEL.IN, async () => {
  return await new Promise((resolve, reject) => {
    childProcess.exec("conda info", (error, stdout, stderr) => {
      return resolve(error === null);
    });
  });
});

ipcMain.handle(FETCH_NEEDS_INSTALL_CHANNEL.IN, async () => {
  return !(await exists(getInstalledPath()));
});

ipcMain.handle(FINISH_INSTALL_CHANNEL.IN, async () => {
  const installedPath = getInstalledPath();
  if (!(await exists(installedPath))) {
    await fsPromises.writeFile(installedPath, "");
  }
});

ipcMain.on(INSTALL_BACKEND_CHANNEL.IN, async (event: IpcMainEvent) => {
  const removeCondaProcess = childProcess.spawn(`conda`, [
    "env",
    "remove",
    "--name",
    CONDA_ENV_NAME,
    "-y",
  ]);
  removeCondaProcess.on("exit", () => {
    const condaInstallerProcess = childProcess.spawn(
      "conda",
      ["env", "create", "-f", "environment.yml"],
      {
        cwd: CONDA_PATH,
      }
    );

    condaInstallerProcess.on("exit", (exitCode: number | null) => {
      const reply: InstallBackendReplyInterface = {
        type: "finished",
        message: "",
        success: exitCode === 0,
      };
      event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
    });

    condaInstallerProcess.stderr.on("data", (data: any) => {
      const reply: InstallBackendReplyInterface = {
        type: "error",
        message: data.toString(),
      };
      event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
    });

    condaInstallerProcess.stdout.on("data", (data: any) => {
      const reply: InstallBackendReplyInterface = {
        type: "message",
        message: data.toString(),
      };
      event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
    });
  });
});
