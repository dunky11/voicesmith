import React, { useEffect, useState, useRef, ReactElement } from "react";
import { IpcRendererEvent } from "electron";
import { Spin } from "antd";
import { createUseStyles } from "react-jss";
import { useSelector } from "react-redux";
import { RootState } from "../../app/store";
import {
  TerminalMessage,
  InstallBackendReplyInterface,
  InstallerOptionsInterface,
} from "../../interfaces";
import NoCloseModal from "../../components/modals/NoCloseModal";
import Terminal from "../../components/log_printer/Terminal";
import {
  INSTALL_BACKEND_CHANNEL,
  START_SERVER_CHANNEL,
  FETCH_NEEDS_INSTALL_CHANNEL,
  FINISH_INSTALL_CHANNEL,
  START_BACKEND_CHANNEL,
} from "../../channels";
import { pingServer } from "../../utils";
import InstallerOptions from "./InstallerOptions";
const { ipcRenderer } = window.require("electron");

const useClasses = createUseStyles({
  wrapper: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
});

export default function MainLoading({
  onServerIsReady,
}: {
  onServerIsReady: () => void;
}): ReactElement {
  const [initStep, setInitStep] = useState<
    | "fetchNeedsInstall"
    | "getInstallOptions"
    | "finishingInstall"
    | "startingBackend"
    | "startingServer"
    | "finished"
    | "installing"
  >("fetchNeedsInstall");
  const [installerMessages, setInstallerMessages] = useState<TerminalMessage[]>(
    []
  );
  const appInfo = useSelector((state: RootState) => state.appInfo);
  const classes = useClasses();
  const installerOptionsRef = useRef<InstallerOptionsInterface>(null);

  const fetchNeedsInstall = () => {
    ipcRenderer
      .invoke(FETCH_NEEDS_INSTALL_CHANNEL.IN)
      .then((needsInstall: boolean) => {
        if (needsInstall) {
          setInitStep("getInstallOptions");
        } else {
          setInitStep("startingBackend");
        }
      });
  };

  const finishInstall = () => {
    ipcRenderer.invoke(FINISH_INSTALL_CHANNEL.IN).then(() => {
      setInitStep("startingBackend");
    });
  };

  const startServer = () => {
    ipcRenderer.invoke(START_SERVER_CHANNEL.IN);
    pingServer(true, () => {
      setInitStep("finished");
    });
  };

  const startBackend = () => {
    ipcRenderer.invoke(START_BACKEND_CHANNEL.IN).then(() => {
      setInitStep("startingServer");
    });
  };

  const installBackend = () => {
    ipcRenderer.removeAllListeners(INSTALL_BACKEND_CHANNEL.REPLY);
    ipcRenderer.on(
      INSTALL_BACKEND_CHANNEL.REPLY,
      (event: IpcRendererEvent, reply: InstallBackendReplyInterface) => {
        switch (reply.type) {
          case "finished": {
            if (reply.success) {
              setInitStep("finishingInstall");
            }
            break;
          }
          case "error": {
            setInstallerMessages((installerMessages) => [
              ...installerMessages,
              { type: "error", message: reply.message },
            ]);
            break;
          }
          case "message": {
            setInstallerMessages((installerMessages) => [
              ...installerMessages,
              { type: "message", message: reply.message },
            ]);
            break;
          }
          default: {
            throw Error(
              `No branch selected in switch-statement '${reply.type}'.`
            );
          }
        }
      }
    );
    ipcRenderer.send(INSTALL_BACKEND_CHANNEL.IN, installerOptionsRef.current);
  };

  const onInstallerOptions = (installerOptions: InstallerOptionsInterface) => {
    installerOptionsRef.current = installerOptions;
    setInitStep("installing");
  };

  useEffect(() => {
    switch (initStep) {
      case "fetchNeedsInstall": {
        fetchNeedsInstall();
        break;
      }
      case "getInstallOptions": {
        break;
      }
      case "installing": {
        installBackend();
        break;
      }
      case "finishingInstall": {
        finishInstall();
        break;
      }
      case "startingBackend": {
        startBackend();
        break;
      }
      case "startingServer": {
        startServer();
        break;
      }
      case "finished": {
        onServerIsReady();
        break;
      }
    }
  }, [initStep]);

  useEffect(() => {
    return () => {
      ipcRenderer.removeAllListeners(INSTALL_BACKEND_CHANNEL.REPLY);
    };
  }, []);

  return (
    <div className={classes.wrapper}>
      <InstallerOptions
        appInfo={appInfo}
        open={initStep === "getInstallOptions"}
        onFinish={onInstallerOptions}
      />
      <NoCloseModal
        title="Installing Backend"
        visible={initStep === "installing"}
      >
        <Terminal messages={installerMessages}></Terminal>
      </NoCloseModal>
      <Spin size="large"></Spin>
    </div>
  );
}
