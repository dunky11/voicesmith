import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, useHistory, Route } from "react-router-dom";
import { Steps, Row, Col, Card, Breadcrumb } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useSelector } from "react-redux";
import BreadcrumbItem from "../../../components/breadcrumb/BreadcrumbItem";
import { RootState } from "../../../app/store";
import {
  RunInterface,
  TextNormalizationRunInterface,
} from "../../../interfaces";
import { useInterval } from "../../../utils";
import { POLL_LOGFILE_INTERVALL } from "../../../config";
import Configuration from "./Configuration";
import Preprocessing from "./Preprocessing";
import ChooseSamples from "./ChooseSamples";
import {
  FETCH_TEXT_NORMALIZATION_RUNS_CHANNEL,
  FETCH_TEXT_NORMALIZATION_RUNS_CHANNEL_TYPES,
} from "../../../channels";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
const { ipcRenderer } = window.require("electron");

const stepToPath: {
  [key: number]: string;
} = {
  0: PREPROCESSING_RUNS_ROUTE.TEXT_NORMALIZATION.CONFIGURATION.ROUTE,
  1: PREPROCESSING_RUNS_ROUTE.TEXT_NORMALIZATION.RUNNING.ROUTE,
  2: PREPROCESSING_RUNS_ROUTE.TEXT_NORMALIZATION.CHOOSE_SAMPLES.ROUTE,
};

const stepToTitle: {
  [key: number]: string;
} = {
  0: "Configuration",
  1: "Normalizing Text",
  2: "Pick Samples",
};

export default function TextNormalization({
  preprocessingRun,
}: {
  preprocessingRun: RunInterface | null;
}): ReactElement {
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const isMounted = useRef(false);
  const [current, setCurrent] = useState(0);
  const history = useHistory();
  const [run, setRun] = useState<TextNormalizationRunInterface | null>(null);

  const selectedIsRunning =
    running !== null &&
    running.type === "textNormalizationRun" &&
    running.ID == preprocessingRun?.ID;

  const fetchRun = () => {
    if (preprocessingRun === null) {
      return;
    }
    const args: FETCH_TEXT_NORMALIZATION_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      ID: preprocessingRun.ID,
    };
    ipcRenderer
      .invoke(FETCH_TEXT_NORMALIZATION_RUNS_CHANNEL.IN, args)
      .then((run: FETCH_TEXT_NORMALIZATION_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        setRun(run[0]);
      });
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

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <BreadcrumbItem to={PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE}>
          Preprocessing Runs
        </BreadcrumbItem>
        {preprocessingRun && (
          <BreadcrumbItem>{preprocessingRun.name}</BreadcrumbItem>
        )}
        <BreadcrumbItem>{stepToTitle[current]}</BreadcrumbItem>
      </Breadcrumb>
      <Row gutter={[0, 100]}>
        <Col className="gutter-row" span={4}>
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
        <Col span={20}>
          <Switch>
            <Route
              render={() =>
                run !== null && (
                  <Configuration onStepChange={onStepChange} run={run} />
                )
              }
              path={stepToPath[0]}
            ></Route>
            <Route
              render={() =>
                run !== null && (
                  <Preprocessing onStepChange={onStepChange} run={run} />
                )
              }
              path={stepToPath[1]}
            ></Route>
            <Route
              render={() =>
                run !== null && (
                  <ChooseSamples onStepChange={onStepChange} run={run} />
                )
              }
              path={stepToPath[2]}
            ></Route>
          </Switch>
        </Col>
      </Row>
    </>
  );
}
