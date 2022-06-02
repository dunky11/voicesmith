import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, useHistory, Route, Link } from "react-router-dom";
import { Steps, Breadcrumb, Row, Col, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import {
  PreprocessingRunInterface,
  RunInterface,
  UsageStatsInterface,
  TextNormalizationInterface,
  SampleSplittingRunInterface,
} from "../../../interfaces";
import { useInterval } from "../../../utils";
import { POLL_LOGFILE_INTERVALL, SERVER_URL } from "../../../config";
import Configuration from "./Configuration";
import Preprocessing from "./Preprocessing";
import ChooseSamples from "./ChooseSamples";
import { FETCH_SAMPLES_SPLITTING_RUNS_CHANNEL } from "../../../channels";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
const { ipcRenderer } = window.require("electron");

const stepToPath: {
  [key: number]: string;
} = {
  0: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.CONFIGURATION.ROUTE,
  1: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.RUNNING.ROUTE,
  2: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.CHOOSE_SAMPLES.ROUTE,
};

const stepToTitle: {
  [key: number]: string;
} = {
  0: "Configuration",
  1: "Splitting Samples",
  2: "Pick Samples",
};

export default function TextNormalization({
  preprocessingRun,
  running,
  continueRun,
  stopRun,
}: {
  preprocessingRun: PreprocessingRunInterface | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
}): ReactElement {
  const isMounted = useRef(false);
  const [current, setCurrent] = useState(0);
  const history = useHistory();
  const [run, setRun] = useState<SampleSplittingRunInterface | null>(null);
  const [usageStats, setUsageStats] = useState<UsageStatsInterface[]>([]);

  const selectedIsRunning =
    running !== null &&
    running.type === "sampleSplittingRun" &&
    running.ID == preprocessingRun?.ID;

  const fetchRun = () => {
    if (preprocessingRun === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_SAMPLES_SPLITTING_RUNS_CHANNEL.IN, preprocessingRun.ID)
      .then((run: SampleSplittingRunInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setRun(run[0]);
      });
  };

  const pollUsageInfo = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", `${SERVER_URL}/get-system-info`);
    ajax.onload = () => {
      if (!isMounted.current) {
        return;
      }
      const response: UsageStatsInterface = JSON.parse(ajax.responseText);
      if (usageStats.length >= 100) {
        usageStats.shift();
      }
      setUsageStats([
        ...usageStats,
        {
          cpuUsage: response["cpuUsage"],
          diskUsed: parseFloat(response["diskUsed"].toFixed(2)),
          totalDisk: parseFloat(response["totalDisk"].toFixed(2)),
          ramUsed: parseFloat(response["ramUsed"].toFixed(2)),
          totalRam: parseFloat(response["totalRam"].toFixed(2)),
        },
      ]);
    };
    ajax.send();
  };

  const onStepChange = (current: number) => {
    history.push(stepToPath[current]);
    setCurrent(current);
  };

  useEffect(() => {
    history.push(stepToPath[current]);
  }, [current]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  useInterval(fetchRun, POLL_LOGFILE_INTERVALL);
  useInterval(pollUsageInfo, 1000);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>
          <Link to={PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE}>
            Preprocessing Runs
          </Link>
        </Breadcrumb.Item>
        {preprocessingRun && (
          <Breadcrumb.Item>{preprocessingRun.name}</Breadcrumb.Item>
        )}
        <Breadcrumb.Item>{stepToTitle[current]}</Breadcrumb.Item>
      </Breadcrumb>
      <Row gutter={[0, 100]}>
        <Col span={4}>
          <Card style={{ borderRight: "none", height: "100%" }}>
            <Steps
              current={current}
              onChange={setCurrent}
              direction="vertical"
              size="small"
            >
              <Steps.Step title={stepToTitle[0]} />
              <Steps.Step
                title={stepToTitle[1]}
                disabled={run === null || ["not_started"].includes(run.stage)}
                icon={selectedIsRunning ? <LoadingOutlined /> : undefined}
              />
              <Steps.Step
                disabled={
                  run === null ||
                  ["not_started", "text_normalization"].includes(run.stage)
                }
                title={stepToTitle[2]}
              />
            </Steps>
          </Card>
        </Col>
        <Col className="gutter-row" span={20}>
          <Switch>
            <Route
              render={() => (
                <Configuration
                  onStepChange={onStepChange}
                  running={running}
                  continueRun={continueRun}
                  run={run}
                />
              )}
              path={stepToPath[0]}
            ></Route>
            <Route
              render={() => (
                <Preprocessing
                  onStepChange={onStepChange}
                  run={run}
                  running={running}
                  continueRun={continueRun}
                  usageStats={usageStats}
                  stopRun={stopRun}
                />
              )}
              path={stepToPath[1]}
            ></Route>
            <Route
              render={() => (
                <ChooseSamples
                  onStepChange={onStepChange}
                  run={run}
                  running={running}
                  continueRun={continueRun}
                  stopRun={stopRun}
                />
              )}
              path={stepToPath[2]}
            ></Route>
          </Switch>
        </Col>
      </Row>
    </>
  );
}
