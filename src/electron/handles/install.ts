import { ipcMain, IpcMainEvent, app } from "electron";
import childProcess from "child_process";
import { ASSETS_PATH, INSTALLED_PATH, POETRY_PATH } from "../utils/globals";
import { exists } from "../utils/files";
import fsNative from "fs";
const fsPromises = fsNative.promises;

ipcMain.handle("fetch-needs-install", async () => {
  return !(await exists(INSTALLED_PATH));
});

ipcMain.handle("install-success", async () => {
  if (!(await exists(INSTALLED_PATH))) {
    await fsPromises.writeFile(INSTALLED_PATH, "");
  }
});

ipcMain.on("install-backend-docker", async (event: IpcMainEvent) => {
  const dockerInstallProcess = childProcess.spawn(
    "docker",
    ["build", "-t", "voice_smith", "."],
    {
      cwd: ASSETS_PATH,
    }
  );

  dockerInstallProcess.stdout.on("data", (data: any) => {
    event.reply("install-backend-reply", {
      type: "message",
      message: data.toString(),
    });
  });

  dockerInstallProcess.stderr.on("data", (data: any) => {
    event.reply("install-backend-reply", {
      type: "error",
      message: data.toString(),
    });
  });

  dockerInstallProcess.on("exit", () => {
    event.reply("install-backend-reply", {
      type: "finishedDocker",
    });
  });
});

ipcMain.on("install-backend-poetry", async (event: IpcMainEvent) => {
  const hasPoetry = await new Promise((resolve, reject) => {
    childProcess.exec("poetry", (error, stdout, stderr) => {
      return resolve(error === null);
    });
  });
  if (!hasPoetry) {
    if (process.platform === "win32") {
      await new Promise((resolve, reject) => {
        childProcess.exec(
          "(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -",
          (error, stdout, stderr) => {
            return resolve(error === null);
          }
        );
      });
    } else {
      await new Promise((resolve, reject) => {
        childProcess.exec(
          "curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -",
          (error, stdout, stderr) => {
            return resolve(error === null);
          }
        );
      });
    }
  }

  const poetryInstallProcess = childProcess.spawn("poetry", ["install"], {
    cwd: POETRY_PATH,
  });

  poetryInstallProcess.on("exit", () => {
    event.reply("install-backend-reply", {
      type: "finishedPoetry",
    });
  });

  poetryInstallProcess.stderr.on("data", (data: any) => {
    event.reply("install-backend-reply", {
      type: "error",
      message: data.toString(),
    });
  });

  poetryInstallProcess.stdout.on("data", (data: any) => {
    event.reply("install-backend-reply", {
      type: "message",
      message: data.toString(),
    });
  });
});
