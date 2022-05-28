import { IpcMainEvent } from "electron";
import { spawn, exec, ChildProcess } from "child_process";
import path from "path";
import {
  getAudioSynthDir,
  BACKEND_PATH,
  DB_PATH,
  getModelsDir,
  PORT,
  ASSETS_PATH,
} from "./globals";
import { DB } from "./db";
import { CONDA_ENV_NAME } from "../../config";

let serverProc: ChildProcess = null;
let pyProc: ChildProcess = null;

const spawnCondaShell = (cmd: string): ChildProcess => {
  return spawn(
    `conda run -n ${CONDA_ENV_NAME} --no-capture-output python ${cmd}`,
    {
      cwd: BACKEND_PATH,
      env: { ...process.env, PYTHONNOUSERSITE: "True" },
      shell: true,
    }
  );
};

export const startRun = (
  event: IpcMainEvent,
  scriptName: string,
  args: string[]
): void => {
  pyProc = spawnCondaShell([scriptName, ...args].join(" "));

  pyProc.on("exit", () => {
    event.reply("continue-run-reply", {
      type: "finishedRun",
    });
  });

  pyProc.stderr.on("data", (data: any) => {
    event.reply("continue-run-reply", {
      type: "error",
      errorMessage: data.toString(),
    });
  });

  event.reply("continue-run-reply", {
    type: "startedRun",
  });
};

export const killServerProc = (): void => {
  if (serverProc === null) {
    return;
  }
  serverProc.kill("SIGKILL");
  serverProc = null;
};

export const killPyProc = (): void => {
  if (pyProc === null) {
    return;
  }
  pyProc.kill("SIGKILL");
  pyProc = null;
};

const selectPort = (): number => {
  return PORT;
};

export const createServerProc = (): void => {
  const port = String(selectPort());
  // Make sure database object is created
  DB.getInstance();
  serverProc = spawnCondaShell(
    `${path.join(
      BACKEND_PATH,
      path.join("server.py")
    )} ${port} ${DB_PATH} ${getAudioSynthDir()} ${getModelsDir()} ${ASSETS_PATH}`
  );
  serverProc.stderr.on("data", (data: any) => {
    throw new Error(data.toString());
  });
  serverProc.stdout.on("data", (data: any) => {
    console.log("Server process stdout: " + data.toString());
  });

  if (serverProc != null) {
    console.log("child process success on port " + port);
  }
};

export const exit = (): void => {
  killServerProc();
  killPyProc();
};
