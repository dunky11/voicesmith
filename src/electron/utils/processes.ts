import { IpcMainEvent } from "electron";
import { spawn, exec, ChildProcess } from "child_process";
import path from "path";
import { stopContainer, spawnCondaCmd } from "./docker";
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
import { CONTINUE_TRAINING_RUN_CHANNEL } from "../../channels";

let serverProc: ChildProcess = null;
let pyProc: ChildProcess = null;

const killLastRun = () => {
  const pid = DB.getInstance().prepare("SELECT pid FROM settings").get().pid;
  if (pid !== null) {
    exec(`kill -15 ${pid}`);
  }
};

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
  args: string[],
  logErr: boolean
): void => {
  console.log([scriptName, ...args].join(" "));
  pyProc = spawnCondaShell([scriptName, ...args].join(" "));
  pyProc.on("exit", () => {
    event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, {
      type: "finishedRun",
    });
  });
  if (logErr) {
    pyProc.stderr.on("data", (data: any) => {
      event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, {
        type: "error",
        errorMessage: data.toString(),
      });
    });
  }
  event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, {
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
  pyProc.kill();
  killLastRun();
  pyProc = null;
};

export const createServerProc = (): void => {
  // Make sure database object is created
  DB.getInstance();
  serverProc = spawnCondaCmd(
    [
      "python",
      "./backend/voice_smith/server.py",
      "--port",
      "80",
      "--db_path",
      DB_PATH,
      "--audio_synth_path",
      getAudioSynthDir(),
      "--models_path",
      getModelsDir(),
      "--assets_path",
      ASSETS_PATH,
    ],
    (data: any) => {
      console.log(`Server process stdout: ${data}`);
    },
    (data: any) => {
      console.log(`Server process stderr: ${data}`);
    },
    (code: number) => {
      if (code !== 0) {
        throw new Error(`Error in server process, status code ${code}`);
      }
    }
  );
  if (serverProc != null) {
    console.log(`child process success on port ${PORT}`);
  }
};

export const exit = (): void => {
  stopContainer();
};
