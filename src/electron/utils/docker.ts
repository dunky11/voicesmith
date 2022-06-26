import childProcess, { ChildProcessWithoutNullStreams } from "child_process";
import path from "path";
import {
  CONDA_ENV_NAME,
  DOCKER_CONTAINER_NAME,
  DOCKER_IMAGE_NAME,
} from "../../config";
import {
  CONDA_PATH,
  DB_PATH,
  PORT,
  ASSETS_PATH,
  UserDataPath,
  RESSOURCES_PATH,
} from "./globals";

export const getHasDocker = async () => {
  return await new Promise((resolve, reject) => {
    childProcess.exec("docker info", (error, stdout, stderr) => {
      return resolve(error === null);
    });
  });
};

export const stopContainer = async (): Promise<void> => {
  return new Promise((resolve, reject) => {
    childProcess.exec(`docker stop ${DOCKER_CONTAINER_NAME}`, () => {
      resolve();
    });
  });
};

export const removeContainer = async (): Promise<void> => {
  return new Promise((resolve, reject) => {
    childProcess.exec(`docker rm ${DOCKER_CONTAINER_NAME}`, () => {
      resolve();
    });
  });
};

export const removeImage = async (): Promise<void> => {
  return new Promise((resolve, reject) => {
    childProcess.exec(`docker image rm ${DOCKER_IMAGE_NAME}`, () => {
      resolve();
    });
  });
};

export const resetDocker = async () => {
  await stopContainer();
  await removeContainer();
  await removeImage();
};

export const spawnCondaCmd = (
  args: string[],
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null,
  onExit: ((code: number) => void) | null
): ChildProcessWithoutNullStreams => {
  console.log(
    ["conda", "run", "-n", CONDA_ENV_NAME, "--no-capture-output", ...args].join(
      " "
    )
  );
  const proc = childProcess.spawn("docker", [
    "exec",
    DOCKER_CONTAINER_NAME,
    "conda",
    "run",
    "-n",
    CONDA_ENV_NAME,
    "--no-capture-output",
    ...args,
  ]);

  if (onData !== null) {
    proc.stdout.on("data", (data: any) => {
      onData(data.toString());
    });
  }

  if (onError !== null) {
    proc.stderr.on("data", (data: any) => {
      onError(data.toString());
    });
  }

  if (onExit !== null) {
    proc.on("exit", onExit);
  }
  return proc;
};

const spawnCondaCmdPromise = async (
  args: string[],
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null
): Promise<void> => {
  return new Promise((resolve, reject) => {
    spawnCondaCmd(args, onData, onError, (code: number) => {
      if (code === 0) {
        resolve();
      } else {
        const errorText = `docker ${[
          "exec",
          DOCKER_CONTAINER_NAME,
          "conda",
          "run",
          "-n",
          CONDA_ENV_NAME,
          ...args,
        ].join(" ")} failed with a status code of ${code}`;
        if (onError !== null) {
          onError(errorText);
        }
        throw new Error(errorText);
      }
    });
  });
};

const spawnDockerCmdPromise = async (
  args: string[],
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null,
  options: childProcess.SpawnOptionsWithoutStdio = {}
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const proc = childProcess.spawn("docker", args, options);

    if (onData !== null) {
      proc.stdout.on("data", (data: any) => {
        onData(data.toString());
      });
    }

    if (onError !== null) {
      proc.stderr.on("data", (data: any) => {
        onError(data.toString());
      });
    }

    proc.on("exit", (code: number) => {
      if (code === 0) {
        resolve();
      } else {
        const errorText = `docker ${args.join(
          " "
        )} failed with a status code of ${code}`;
        if (onError !== null) {
          onError(errorText);
        }
        throw new Error(errorText);
      }
    });
  });
};

export const startContainer = async (
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null
): Promise<void> => {
  await spawnDockerCmdPromise(
    ["start", DOCKER_CONTAINER_NAME],
    onData,
    onError
  );
};

export const createContainer = async (
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null,
  withGPU: boolean
): Promise<void> => {
  await spawnDockerCmdPromise(
    [
      "run",
      "-itd",
      "--name",
      DOCKER_CONTAINER_NAME,
      "--mount",
      `type=bind,source=${CONDA_PATH},target=/home/backend`,
      "--mount",
      `type=bind,source=${ASSETS_PATH},target=/home/assets`,
      "--mount",
      `type=bind,source=${path.dirname(DB_PATH)},target=/home/db`,
      "--mount",
      `type=bind,source=${UserDataPath().getPath()},target=/home/data`,
      "--ulimit",
      "stack=67108864",
      "-p",
      `${PORT}:80`,
      "-u",
      "$(id -u):$(id -g)",
      ...(withGPU ? ["--gpus", "all"] : []),
      DOCKER_IMAGE_NAME,
    ],
    onData,
    onError
  );
};

export const installEnvironment = async (
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null
): Promise<void> => {
  await spawnDockerCmdPromise(
    [
      "exec",
      DOCKER_CONTAINER_NAME,
      "conda",
      "env",
      "update",
      "-f",
      "./backend/environment.yml",
      "--prune",
    ],
    onData,
    onError
  );
  await spawnDockerCmdPromise(
    ["exec", DOCKER_CONTAINER_NAME, "conda", "init", "bash"],
    onData,
    onError
  );
  await spawnCondaCmdPromise(
    ["mfa", "models", "download", "acoustic", "english_us_arpa"],
    onData,
    onError
  );
};
