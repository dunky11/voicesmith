import { IpcMainEvent } from "electron";
import childProcess from "child_process";
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

let serverProc: any = null;
let pyProc: any = null;

export const startRun = (
  event: IpcMainEvent,
  scriptName: string,
  args: string[]
) => {
  pyProc = childProcess.spawn(
    "poetry",
    ["run", "python", scriptName, ...args],
    { cwd: BACKEND_PATH }
  );

  pyProc.on("exit", () => {
    event.reply("continue-run-reply", {
      type: "finishedRun",
    });
  });

  pyProc.stderr.on("data", (data: any) => {
    console.log(data.toString());
    event.reply("continue-run-reply", {
      type: "error",
      errorMessage: data.toString(),
    });
  });

  event.reply("continue-run-reply", {
    type: "startedRun",
  });
};

export const killServerProc = () => {
  if (serverProc === null) {
    return;
  }
  serverProc.kill("SIGKILL");
  serverProc = null;
};

export const killPyProc = () => {
  if (pyProc === null) {
    return;
  }
  pyProc.kill("SIGKILL");
  pyProc = null;
};

const selectPort = () => {
  return PORT;
};

export const createServerProc = () => {
  const port = String(selectPort());
  // Make sure database object is created
  DB.getInstance();
  console.log(BACKEND_PATH);
  serverProc = childProcess.spawn(
    `conda run -n ${CONDA_ENV_NAME} --no-capture-output python ${path.join(
      BACKEND_PATH,
      path.join("server.py")
    )} ${port} ${DB_PATH} ${getAudioSynthDir()} ${getModelsDir()} ${ASSETS_PATH}`,
    {
      cwd: BACKEND_PATH,
      env: { ...process.env, PYTHONNOUSERSITE: "True" },
      shell: true,
    }
  );
  serverProc.stdout.on("data", (data: any) => {
    throw new Error(data.toString());
  });
  serverProc.stderr.on("data", (data: any) => {
    console.log("Server process stdout: " + data.toString());
  });

  if (serverProc != null) {
    console.log("child process success on port " + port);
  }
};

const killDockerProc = () => {
  childProcess.exec("docker kill voice_smith");
};

export const exit = () => {
  killServerProc();
  killPyProc();
  killDockerProc();
};
