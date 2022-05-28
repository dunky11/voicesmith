import React, { useEffect, useState, useRef } from "react";
import { IpcRendererEvent } from "electron";
import { Spin } from "antd";
import { createUseStyles } from "react-jss";
import {
  TerminalMessage,
  InstallBackendReplyInterface,
} from "../../interfaces";
import NoCloseModal from "../../components/modals/NoCloseModal";
import Terminal from "../../components/log_printer/Terminal";
import {
  INSTALL_BACKEND_CHANNEL,
  START_SERVER_CHANNEL,
  FETCH_HAS_CONDA_CHANNEL,
  FETCH_NEEDS_INSTALL_CHANNEL,
  FINISH_INSTALL_CHANNEL,
} from "../../channels";
import { pingServer } from "../../utils";
const { ipcRenderer, shell } = window.require("electron");

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
}) {
  const [initStep, setInitStep] = useState<
    | "fetchConda"
    | "noConda"
    | "fetchNeedsInstall"
    | "installing"
    | "finishingInstall"
    | "startingServer"
    | "finished"
  >("fetchConda");
  const [installerMessages, setInstallerMessages] = useState<TerminalMessage[]>(
    []
  );
  const classes = useClasses();

  const fetchNeedsInstall = () => {
    ipcRenderer
      .invoke(FETCH_NEEDS_INSTALL_CHANNEL.IN)
      .then((needsInstall: boolean) => {
        if (needsInstall) {
          setInitStep("installing");
        } else {
          setInitStep("startingServer");
        }
      });
  };

  const fetchHasConda = () => {
    ipcRenderer.invoke(FETCH_HAS_CONDA_CHANNEL.IN).then((hasConda: boolean) => {
      if (hasConda) {
        setInitStep("fetchNeedsInstall");
      } else {
        setInitStep("noConda");
      }
    });
  };

  const finishInstall = () => {
    ipcRenderer.invoke(FINISH_INSTALL_CHANNEL.IN).then(() => {
      setInitStep("startingServer");
    });
  };

  const startServer = () => {
    ipcRenderer.invoke(START_SERVER_CHANNEL.IN);
    pingServer(true, () => {
      setInitStep("finished");
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
    ipcRenderer.send(INSTALL_BACKEND_CHANNEL.IN);
  };

  useEffect(() => {
    switch (initStep) {
      case "fetchConda": {
        fetchHasConda();
        break;
      }
      case "fetchNeedsInstall": {
        fetchNeedsInstall();
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
      <NoCloseModal
        title="Anaconda was not found on your system"
        visible={initStep === "noConda"}
      >
        <p>
          There has been no installation of Anaconda detected on your system.
          Please navigate to{" "}
          <a
            onClick={() => {
              shell.openExternal("https://www.anaconda.com/");
            }}
          >
            https://www.anaconda.com/
          </a>{" "}
          or{" "}
          <a
            onClick={() => {
              shell.openExternal(
                "https://docs.conda.io/en/latest/miniconda.html"
              );
            }}
          >
            https://docs.conda.io/en/latest/miniconda.html
          </a>{" "}
          to download and install either Anaconda or Miniconda. Afterwards,
          please restart this application.
        </p>
      </NoCloseModal>
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
