import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, useHistory, Route, Link } from "react-router-dom";
import { Steps, Breadcrumb, Row, Col, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useSelector } from "react-redux";
import { RootState } from "../../../app/store";
import { RunInterface, SampleSplittingRunInterface } from "../../../interfaces";
import { useInterval } from "../../../utils";
import { POLL_LOGFILE_INTERVALL } from "../../../config";
import Configuration from "./Configuration";
import Preprocessing from "./Preprocessing";
import ChooseSamples from "./ChooseSamples";
import ApplyChanges from "./ApplyChanges";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import {
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES,
} from "../../../channels";

const { ipcRenderer } = window.require("electron");

const stepToPath: {
  [key: number]: string;
} = {
  0: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.CONFIGURATION.ROUTE,
  1: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.RUNNING.ROUTE,
  2: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.CHOOSE_SAMPLES.ROUTE,
  3: PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.APPLY_CHANGES.ROUTE,
};

const stepToTitle: {
  [key: number]: string;
} = {
  0: "Configuration",
  1: "Splitting Samples",
  2: "Pick Samples",
  3: "Apply Changes",
};

export default function SampleSplitting({
  preprocessingRun,
}: {
  preprocessingRun: RunInterface;
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
  const [run, setRun] = useState<SampleSplittingRunInterface | null>(null);

  const selectedIsRunning =
    running !== null &&
    running.type === "sampleSplittingRun" &&
    running.ID === preprocessingRun.ID;

  const fetchRun = () => {
    const args: FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      ID: preprocessingRun.ID,
    };
    ipcRenderer
      .invoke(FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL.IN, args)
      .then((run: FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
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
        <Breadcrumb.Item>
          <Link to={PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE}>
            Preprocessing Runs
          </Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>{preprocessingRun.name}</Breadcrumb.Item>
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
                icon={
                  selectedIsRunning &&
                  run !== null &&
                  [
                    "copying_files",
                    "gen_vocab",
                    "gen_alignments",
                    "creating_splits",
                  ].includes(run.stage) ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                disabled={
                  run === null ||
                  [
                    "not_started",
                    "copying_files",
                    "gen_vocab",
                    "gen_alignments",
                    "creating_splits",
                  ].includes(run.stage)
                }
                title={stepToTitle[2]}
              />
              <Steps.Step
                disabled={
                  run === null ||
                  [
                    "not_started",
                    "copying_files",
                    "gen_vocab",
                    "gen_alignments",
                    "creating_splits",
                    "choose_samples",
                  ].includes(run.stage)
                }
                title={stepToTitle[3]}
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
            <Route
              render={() =>
                run !== null && (
                  <ApplyChanges onStepChange={onStepChange} run={run} />
                )
              }
              path={stepToPath[3]}
            ></Route>
          </Switch>
        </Col>
      </Row>
    </>
  );
}
