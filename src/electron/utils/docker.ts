import childProcess, { ChildProcessWithoutNullStreams } from "child_process";
import {
  CONDA_ENV_NAME,
  DOCKER_CONTAINER_NAME,
  DOCKER_IMAGE_NAME,
} from "../../config";
import { CONDA_PATH, PORT } from "./globals";

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
  const proc = childProcess.spawn("docker", [
    "exec",
    DOCKER_CONTAINER_NAME,
    "conda",
    "run",
    "-n",
    CONDA_ENV_NAME,
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

const spawnDockerCmdPromise = async (
  args: string[],
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const proc = childProcess.spawn("docker", args);

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
        onError(errorText);
        throw new Error(errorText);
      }
    });
  });
};

export const buildImage = async (
  onData: ((data: string) => void) | null,
  onError: ((data: string) => void) | null
): Promise<void> => {
  await spawnDockerCmdPromise(
    ["build", ".", "--rm", "-t", DOCKER_IMAGE_NAME],
    onData,
    onError
  );
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
      "voice_smith",
      "--mount",
      `type=bind,source=${CONDA_PATH},target=/home/backend`,
      "--ulimit",
      "stack=67108864",
      "-p",
      `${PORT}:80`,
      ...(withGPU ? ["--gpus", "all"] : []),
      DOCKER_CONTAINER_NAME,
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
      "create",
      "-f",
      "./backend/environment.yml",
    ],
    onData,
    onError
  );
};
