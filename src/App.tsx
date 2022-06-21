import React, { useRef, useEffect, useState, ReactElement } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import { Layout, Menu, Typography, notification, Divider } from "antd";
import {
  ShareAltOutlined,
  FundFilled,
  DatabaseOutlined,
  ClearOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import { createUseStyles } from "react-jss";
import { useDispatch, useSelector } from "react-redux";
import { editAppInfo } from "./features/appInfoSlice";
import { RootState } from "./app/store";
import MainLoading from "./pages/main_loading/MainLoading";
import Models from "./pages/models/Models";
import Synthesize from "./pages/models/Synthesize";
import TrainingRuns from "./pages/training_runs/TrainingRuns";
import Datasets from "./pages/datasets/Datasets";
import PreprocessingRuns from "./pages/preprocessing_runs/PreprocessingRuns";
import { AppInfoInterface } from "./interfaces";
import { pingServer } from "./utils";
import {
  DATASETS_ROUTE,
  MODELS_ROUTE,
  PREPROCESSING_RUNS_ROUTE,
  SETTINGS_ROUTE,
  TRAINING_RUNS_ROUTE,
  RUN_QUEUE_ROUTE,
} from "./routes";
import Settings from "./pages/settings/Settings";
import RunQueue from "./pages/run_queue/RunQueue";
import {
  CONTINUE_TRAINING_RUN_CHANNEL,
  GET_APP_INFO_CHANNEL,
} from "./channels";
import RunManager from "./components/run_management/RunManager";

const { ipcRenderer } = window.require("electron");

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
    minHeight: "100vh",
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
  leftLayout: { minHeight: "100%" },
  contentLayout: {},
  content: { margin: "24px !important" },
});

export default function App(): ReactElement {
  const classes = useStyles();
  const history = useHistory();
  const isMounted = useRef(false);
  const dispatch = useDispatch();
  const appInfo = useSelector((state: RootState) => state.appInfo);
  const [selectedKeys, setSelectedKeys] = useState<string[]>(["models"]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [navIsDisabled, setNavIsDisabled] = useState(false);
  const [serverIsReady, setServerIsReady] = useState(false);

  const fetchAppInfo = () => {
    ipcRenderer
      .invoke(GET_APP_INFO_CHANNEL.IN)
      .then((appInfo: AppInfoInterface) => {
        dispatch(editAppInfo(appInfo));
      });
  };

  const onModelSelect = (model: any) => {
    history.push(MODELS_ROUTE.SYNTHESIZE.ROUTE);
    setSelectedModel(model);
  };

  const onNavigationSelect = ({
    key,
  }: {
    key:
      | "models"
      | "datasets"
      | "training-runs"
      | "preprocessing-runs"
      | "settings"
      | "run-queue";
  }) => {
    setSelectedKeys([key]);
    switch (key) {
      case "models":
        history.push(MODELS_ROUTE.SELECTION.ROUTE);
        break;
      case "datasets":
        history.push(DATASETS_ROUTE.SELECTION.ROUTE);
        break;
      case "training-runs":
        history.push(TRAINING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
        break;
      case "preprocessing-runs":
        history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
        break;
      case "settings":
        history.push(SETTINGS_ROUTE.ROUTE);
        break;
      case "run-queue":
        history.push(RUN_QUEUE_ROUTE.ROUTE);
        break;
      default:
        throw new Error(
          `No case selected in switch-statement: '${key}' is not a valid key.`
        );
    }
  };

  const onServerIsReady = () => {
    pushRoute(MODELS_ROUTE.SELECTION.ROUTE);
    setServerIsReady(true);
  };

  const pushRoute = (route: string) => {
    history.push(route);
    if (route.includes(MODELS_ROUTE.ROUTE)) {
      setSelectedKeys(["models"]);
    } else if (route.includes(DATASETS_ROUTE.ROUTE)) {
      setSelectedKeys(["datasets"]);
    } else if (route.includes(PREPROCESSING_RUNS_ROUTE.ROUTE)) {
      setSelectedKeys(["preprocessing-runs"]);
    } else if (route.includes(SETTINGS_ROUTE.ROUTE)) {
      setSelectedKeys(["settings"]);
    } else if (route.includes(RUN_QUEUE_ROUTE.ROUTE)) {
      setSelectedKeys(["run-queue"]);
    } else {
      throw new Error(`Route '${route}' is not a valid route.`);
    }
  };

  useEffect(() => {
    isMounted.current = true;
    fetchAppInfo();
    pingServer(false, onServerIsReady);
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners(CONTINUE_TRAINING_RUN_CHANNEL.REPLY);
    };
  }, []);

  return (
    <>
      <RunManager />
      <Layout className={classes.leftLayout}>
        <Layout.Sider className={classes.sider}>
          <div className={classes.logoWrapper}>
            <Typography.Title level={3} className={classes.logo}>
              VOICESMITH
            </Typography.Title>
            <Typography.Text className={classes.logoVersion}>
              {appInfo === null ? "" : `v${appInfo.version}`}
            </Typography.Text>
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
                  onNavigationSelect({ key: "run-queue" });
                }}
                key="run-queue"
                icon={<ClearOutlined className={classes.navIcon} />}
                className={classes.navItem}
              >
                Run Queue
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
                      path={MODELS_ROUTE.SELECTION.ROUTE}
                    ></Route>
                    <Route
                      render={(props) => (
                        <Synthesize {...props} selectedModel={selectedModel} />
                      )}
                      path={MODELS_ROUTE.SYNTHESIZE.ROUTE}
                    ></Route>
                  </Switch>
                )}
                path={MODELS_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => <TrainingRuns></TrainingRuns>}
                path={TRAINING_RUNS_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => <Datasets></Datasets>}
                path={DATASETS_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => <PreprocessingRuns></PreprocessingRuns>}
                path={PREPROCESSING_RUNS_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => <RunQueue></RunQueue>}
                path={RUN_QUEUE_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => (
                  <Settings setNavIsDisabled={setNavIsDisabled}></Settings>
                )}
                path={SETTINGS_ROUTE.ROUTE}
              ></Route>
              <Route
                render={() => (
                  <MainLoading onServerIsReady={onServerIsReady}></MainLoading>
                )}
              ></Route>
            </Switch>
          </Layout.Content>
        </Layout>
      </Layout>
    </>
  );
}
