import { ipcMain, IpcMainEvent } from "electron";
import { getInstalledPath } from "../utils/globals";
import {
  INSTALL_BACKEND_CHANNEL,
  FINISH_INSTALL_CHANNEL,
  FETCH_HAS_DOCKER_CHANNEL,
  FETCH_NEEDS_INSTALL_CHANNEL,
} from "../../channels";
import { exists } from "../utils/files";
import fsNative from "fs";
import {
  InstallBackendReplyInterface,
  InstallerOptionsInterface,
} from "../../interfaces";
import {
  getHasDocker,
  buildImage,
  resetDocker,
  createContainer,
  installEnvironment,
  stopContainer,
} from "../utils/docker";
const fsPromises = fsNative.promises;

ipcMain.handle(FETCH_HAS_DOCKER_CHANNEL.IN, async () => {
  return await getHasDocker();
});

ipcMain.handle(FETCH_NEEDS_INSTALL_CHANNEL.IN, async () => {
  return !(await exists(getInstalledPath()));
});

ipcMain.handle(FINISH_INSTALL_CHANNEL.IN, async () => {
  const installedPath = getInstalledPath();
  console.log(installedPath);
  console.log(installedPath);
  console.log(installedPath);
  console.log(installedPath);
  if (!(await exists(installedPath))) {
    console.log("WRITING FILE");
    console.log("WRITING FILE");
    console.log("WRITING FILE");
    console.log("WRITING FILE");
    await fsPromises.writeFile(installedPath, "");
  }
});

ipcMain.on(
  INSTALL_BACKEND_CHANNEL.IN,
  async (event: IpcMainEvent, installerOptions: InstallerOptionsInterface) => {
    await resetDocker();
    const onData = (data: string) => {
      const reply: InstallBackendReplyInterface = {
        type: "message",
        message: data,
      };
      event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
    };
    const onError = (data: string) => {
      const reply: InstallBackendReplyInterface = {
        type: "error",
        message: data,
      };
      event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
    };
    await buildImage(onData, onError);
    await createContainer(onData, onError, installerOptions.device === "GPU");
    await installEnvironment(onData, onError);
    const reply: InstallBackendReplyInterface = {
      type: "finished",
      message: "",
      success: true,
    };
    await stopContainer();
    event.reply(INSTALL_BACKEND_CHANNEL.REPLY, reply);
  }
);
