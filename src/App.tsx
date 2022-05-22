import React, { useRef, useEffect, useState } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import { Layout, Menu, Typography, notification, Divider, Modal } from "antd";
import {
  ShareAltOutlined,
  FundFilled,
  DatabaseOutlined,
  ClearOutlined,
  SettingOutlined,
  FormOutlined,
} from "@ant-design/icons";
import { createUseStyles } from "react-jss";
import MainLoading from "./pages/main_loading/MainLoading";
import Models from "./pages/models/Models";
import Synthesize from "./pages/models/Synthesize";
import TrainingRuns from "./pages/training_runs/TrainingRuns";
import { SERVER_URL, APP_VERSION } from "./config";
import Datasets from "./pages/datasets/Datasets";
import PreprocessingRuns from "./pages/preprocessing_runs/PreprocessingRuns";
import Terminal from "./components/log_printer/Terminal";
import { RunInterface, TerminalMessage } from "./interfaces";
import Settings from "./pages/settings/Settings";
import Transcribe from "./pages/transcribe/Transcribe";
const { ipcRenderer, shell } = window.require("electron");

const useStyles = createUseStyles({
  logoWrapper: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 24,
  },
  logo: {
    color: "#fff !important",
    marginBottom: "0px !important",
    fontSize: "20px !important",
    marginTop: 6,
    fontFamily: "atmospheric",
  },
  logoVersion: { color: "#fff !important", fontWeight: "bold", fontSize: 12 },
  sider: {
    overflow: "auto",
    height: "100vh",
    position: "fixed",
    left: 0,
    top: 0,
    bottom: 0,
    background: "#161619 !important",
  },
  navMenu: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    background: "#161619 !important",
  },
  navItem: {
    borderRadius: 4,
    fontWeight: "bold",
  },
  navIcon: { marginRight: 4, fontSize: 16 },
  divider: { borderColor: "#27272A" },
  leftLayout: { minHeight: "100vh" },
  contentLayout: { marginLeft: 200 },
  content: { margin: "24px !important" },
});

export default function App() {
  const classes = useStyles();
  const history = useHistory();
  const isMounted = useRef(false);
  const [selectedKeys, setSelectedKeys] = useState<string[]>(["models"]);
  const [hasDocker, setHasDocker] = useState(false);
  const [downloadDockerIsOpen, setDownloadDockerIsOpen] = useState(false);
  const [serverIsReady, setServerIsReady] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [navIsDisabled, setNavIsDisabled] = useState(false);
  const onModelSelect = (model: any) => {
    history.push("/models/synthesize");
    setSelectedModel(model);
  };
  const [running, setRunning] = useState<RunInterface | null>(null);
  const [installerIsOpen, setInstallerIsOpen] = useState(false);
  const [installerMessages, setInstallerMessages] = useState<TerminalMessage[]>(
    []
  );
  const hadInstallerError = useRef(false);

  const onNavigationSelect = ({ key }: { key: string }) => {
    setSelectedKeys([key]);
    switch (key) {
      case "models":
        history.push("/models/selection");
        break;
      case "transcribe":
        history.push("/transcribe");
        break;
      case "datasets":
        history.push("/datasets/dataset-selection");
        break;
      case "training-runs":
        history.push("/training-runs/run-selection");
        break;
      case "preprocessing-runs":
        history.push("/preprocessing-runs/run-selection");
        break;
      case "settings":
        history.push("/settings");
        break;
      default:
        throw new Error(
          `No case selected in switch-statement: '${key}' is not a valid key.`
        );
    }
  };

  const pushRoute = (route: string) => {
    history.push(route);
    if (route.includes("/models")) {
      setSelectedKeys(["models"]);
    } else if (route.includes("/datasets")) {
      setSelectedKeys(["datasets"]);
    } else if (route.includes("/preprocessing-runs")) {
      setSelectedKeys(["preprocessing-runs"]);
    } else {
      throw new Error(`Route '${route}' is not a valid route.`);
    }
  };

  const continueRun = (run: RunInterface) => {
    if (running !== null) {
      notification["warning"]({
        message: `Another run is currently active, please stop it before starting this one.`,
        placement: "top",
      });
      return;
    }
    ipcRenderer.removeAllListeners("continue-run-reply");
    ipcRenderer.on(
      "continue-run-reply",
      (
        _: any,
        message: {
          type: string;
          errorMessage?: string;
        }
      ) => {
        if (!isMounted.current) {
          return;
        }

        switch (message.type) {
          case "notEnoughSpeakers": {
            notification["error"]({
              message: "Couldn't Start Run",
              description:
                "This runs dataset contains only one speaker, but it has to have at least two speakers in order to detect potentially noisy samples.",
              placement: "top",
            });
            return;
          }
          case "notEnoughSamples":
            notification["error"]({
              message: "Couldn't Start Run",
              description:
                "This runs dataset contains no samples. Please attach samples to the speakers in your dataset and try again.",
              placement: "top",
            });
            return;
          case "startedRun":
            setRunning(run);
            return;
          case "finishedRun":
            setRunning(null);
            return;
          case "error":
            setRunning(null);
            notification["error"]({
              message: "Oops, an error occured, check logs for more info ...",
              description: message.errorMessage,
              placement: "top",
            });
            return;
          default:
            throw new Error(
              `No branch selected in switch-statement, '${message.type}' is not a valid case ...`
            );
        }
      }
    );
    switch (run.type) {
      case "trainingRun":
        ipcRenderer.send("continue-training-run", run.ID);
        break;
      case "dSCleaning":
        ipcRenderer.send("continue-cleaning-run", run.ID);
        break;
      case "textNormalizationRun":
        ipcRenderer.send("continue-text-normalization-run", run.ID);
        break;
      default:
        throw new Error(
          `No branch selected in switch-statement, '${run.type}' is not a valid case ...`
        );
    }
  };

  const stopRun = () => {
    ipcRenderer.invoke("stop-run").then(() => {
      if (!isMounted.current) {
        return;
      }
      setRunning(null);
    });
  };

  const pingServer = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", SERVER_URL);
    ajax.onload = () => {
      setServerIsReady(true);
    };
    ajax.onerror = () => {
      pingServer();
    };
    ajax.send();
  };

  const fetchHasDocker = () => {
    ipcRenderer.invoke("fetch-has-docker").then((hasDocker: boolean) => {
      if (hasDocker) {
        initiateInstall();
      } else {
        setDownloadDockerIsOpen(true);
      }
      setHasDocker(hasDocker);
    });
  };

  const startServer = () => {
    ipcRenderer.invoke("start-server");
  };

  const initiateInstall = () => {
    ipcRenderer.invoke("fetch-needs-install").then((needsInstall: boolean) => {
      if (needsInstall) {
        installBackend();
      } else {
        startServer();
      }
    });
  };

  const installBackend = () => {
    ipcRenderer.removeAllListeners("install-backend-reply");
    setInstallerIsOpen(true);
    ipcRenderer.on("install-backend-reply", (event: any, reply: any) => {
      switch (reply.type) {
        case "finishedPoetry": {
          if (!hadInstallerError.current) {
            ipcRenderer.invoke("install-success").then(() => {
              setInstallerIsOpen(false);
              startServer();
            });
          }
          break;
        }
        case "finishedDocker": {
          if (!hadInstallerError.current) {
            ipcRenderer.send("install-backend-poetry");
          }
          break;
        }
        case "error": {
          hadInstallerError.current = true;
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
    });
    ipcRenderer.send("install-backend-docker");
  };

  useEffect(() => {
    if (serverIsReady && hasDocker) {
      history.push("/models/selection");
    }
  }, [serverIsReady, hasDocker]);

  useEffect(() => {
    isMounted.current = true;
    pingServer();
    fetchHasDocker();
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners("continue-run-reply");
      ipcRenderer.removeAllListeners("install-backend-reply");
    };
  }, []);

  return (
    <>
      <Modal
        title="Docker was not found on your system"
        visible={downloadDockerIsOpen}
        onOk={null}
        onCancel={null}
        // TODO find way to remove cursor on close icon area hover
        closeIcon={<></>}
        footer={null}
      >
        <p>
          There has been no installation of Docker detected on your system.
          Please navigate to{" "}
          <a
            onClick={() => {
              shell.openExternal("https://www.docker.com/get-started/");
            }}
          >
            https://www.docker.com/get-started/
          </a>{" "}
          to download and install Docker. Afterwards, restart this application.
        </p>
      </Modal>
      <Modal
        title="Installing Backend"
        visible={installerIsOpen}
        onOk={null}
        onCancel={null}
        // TODO find way to remove cursor on close icon area hover
        closeIcon={<></>}
        footer={null}
      >
        <Terminal messages={installerMessages}></Terminal>
      </Modal>
      <Layout className={classes.leftLayout}>
        <Layout.Sider className={classes.sider}>
          <div className={classes.logoWrapper}>
            <Typography.Title level={3} className={classes.logo}>
              VOICESMITH
            </Typography.Title>
            <Typography.Text
              className={classes.logoVersion}
            >{`v${APP_VERSION}`}</Typography.Text>
          </div>
          <div style={{ paddingLeft: 8, paddingRight: 8 }}>
            <Divider className={classes.divider} />
            <Menu
              theme="dark"
              selectedKeys={selectedKeys}
              mode="inline"
              className={classes.navMenu}
              disabled={!serverIsReady || navIsDisabled}
            >
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "models" });
                }}
                key="models"
                icon={<ShareAltOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Models
              </Menu.Item>
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "transcribe" });
                }}
                key="transcribe"
                icon={<FormOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Transcribe
              </Menu.Item>
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "datasets" });
                }}
                key="datasets"
                icon={<DatabaseOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Datasets
              </Menu.Item>
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "training-runs" });
                }}
                key="training-runs"
                icon={<FundFilled className={classes.navIcon} />}
                className={classes.navItem}
              >
                Training Runs
              </Menu.Item>
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "preprocessing-runs" });
                }}
                key="preprocessing-runs"
                icon={<ClearOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Preprocessing
              </Menu.Item>
              <Menu.Item
                onClick={() => {
                  onNavigationSelect({ key: "settings" });
                }}
                key="settings"
                icon={<SettingOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Settings
              </Menu.Item>
            </Menu>
          </div>
        </Layout.Sider>
        <Layout className={classes.contentLayout}>
          <Layout.Content className={classes.content}>
            <Switch>
              <Route
                render={(props) => (
                  <Switch {...props}>
                    <Route
                      render={(props) => (
                        <Models
                          {...props}
                          onModelSelect={onModelSelect}
                          pushRoute={pushRoute}
                        />
                      )}
                      path="/models/selection"
                    ></Route>
                    <Route
                      render={(props) => (
                        <Synthesize {...props} selectedModel={selectedModel} />
                      )}
                      path="/models/synthesize"
                    ></Route>
                  </Switch>
                )}
                path="/models"
              ></Route>
              <Route
                render={() => <Transcribe></Transcribe>}
                path="/transcribe"
              ></Route>
              <Route
                render={() => (
                  <TrainingRuns
                    running={running}
                    continueRun={continueRun}
                    stopRun={stopRun}
                  ></TrainingRuns>
                )}
                path="/training-runs"
              ></Route>
              <Route
                render={() => <Datasets></Datasets>}
                path="/datasets"
              ></Route>
              <Route
                render={() => (
                  <PreprocessingRuns
                    running={running}
                    continueRun={continueRun}
                    stopRun={stopRun}
                  ></PreprocessingRuns>
                )}
                path="/preprocessing-runs"
              ></Route>
              <Route
                render={() => (
                  <Settings
                    running={running}
                    setNavIsDisabled={setNavIsDisabled}
                  ></Settings>
                )}
                path="/settings"
              ></Route>
              <Route render={() => <MainLoading></MainLoading>}></Route>
            </Switch>
          </Layout.Content>
        </Layout>
      </Layout>
    </>
  );
}
