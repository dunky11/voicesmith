import { IpcMainEvent } from "electron";
import childProcess from "child_process";
import {
  AUDIO_SYNTH_DIR,
  BACKEND_PATH,
  DB_PATH,
  MODELS_DIR,
  PORT,
  ASSETS_PATH,
} from "./globals";
import { DB } from "./db";

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
  serverProc = childProcess.spawn(
    "poetry",
    [
      "run",
      "python",
      "server.py",
      port,
      DB_PATH,
      AUDIO_SYNTH_DIR,
      MODELS_DIR,
      ASSETS_PATH,
    ],
    { cwd: BACKEND_PATH }
  );
  serverProc.stderr.on("data", (data: string) => {
    console.log("stderr: " + data);
  });
  serverProc.stdout.on("data", (data: string) => {
    console.log("stdout: " + data);
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
