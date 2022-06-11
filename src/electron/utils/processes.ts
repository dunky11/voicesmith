import { IpcMainEvent } from "electron";
import { exec, ChildProcess } from "child_process";
import { stopContainer, spawnCondaCmd } from "./docker";
import { PORT } from "./globals";
import { DB } from "./db";
import { CONTINUE_TRAINING_RUN_CHANNEL } from "../../channels";
import { DOCKER_CONTAINER_NAME } from "../../config";

let serverProc: ChildProcess = null;
let pyProc: ChildProcess = null;

const killLastRun = () => {
  const pid = DB.getInstance().prepare("SELECT pid FROM settings").get().pid;
  if (pid !== null) {
    exec(`docker exec ${DOCKER_CONTAINER_NAME} kill -15 ${pid}`);
  }
};

export const startRun = (
  event: IpcMainEvent,
  scriptName: string,
  args: string[],
  logErr: boolean
): void => {
  pyProc = spawnCondaCmd(
    ["python", scriptName, ...args],
    null,
    logErr
      ? (data: string) => {
          event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, {
            type: "error",
            errorMessage: data,
          });
        }
      : null,
    (code: number) => {
      event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, {
        type: "finishedRun",
      });
    }
  );

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
    ["python", "./backend/voice_smith/server.py", "--port", "80"],
    null,
    (data: string) => {
      console.log(`stderr in server process: ${data}`);
    },
    (code: number) => {
      if (code !== 0) {
        console.log(
          `Non zero exit code in server process, status code ${code}`
        );
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
