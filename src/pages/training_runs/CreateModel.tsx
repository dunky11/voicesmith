import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, useHistory, Route } from "react-router-dom";
import { Steps, Breadcrumb, Row, Col, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useSelector } from "react-redux";
import BreadcrumbItem from "../../components/breadcrumb/BreadcrumbItem";
import AcousticModelFinetuning from "./AcousticModelFinetuning";
import Configuration from "./Configuration";
import Preprocessing from "./Preprocessing";
import GroundTruthAlignment from "./GroundTruthAlignment";
import VocoderFineTuning from "./VocoderFineTuning";
import SaveModel from "./SaveModel";
import {
  FETCH_TRAINING_RUNS_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL_TYPES,
} from "../../channels";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { getProgressTitle, useInterval } from "../../utils";
import { POLL_LOGFILE_INTERVALL } from "../../config";
import { TRAINING_RUNS_ROUTE } from "../../routes";
import { RootState } from "../../app/store";
const { ipcRenderer } = window.require("electron");

const stepToPath: {
  [key: number]: string;
} = {
  0: TRAINING_RUNS_ROUTE.CREATE_MODEL.CONFIGURATION.ROUTE,
  1: TRAINING_RUNS_ROUTE.CREATE_MODEL.DATA_PREPROCESSING.ROUTE,
  2: TRAINING_RUNS_ROUTE.CREATE_MODEL.ACOUSTIC_TRAINING.ROUTE,
  3: TRAINING_RUNS_ROUTE.CREATE_MODEL.GENERATE_GTA.ROUTE,
  4: TRAINING_RUNS_ROUTE.CREATE_MODEL.VOCODER_TRAINING.ROUTE,
  5: TRAINING_RUNS_ROUTE.CREATE_MODEL.SAVE_MODEL.ROUTE,
};

const stepToTitle: {
  [key: number]: string;
} = {
  0: "Configuration",
  1: "Data Preprocessing",
  2: "Acoustic Model Fine-Tuning",
  3: "Generate Ground Truth Alignments",
  4: "Vocoder Fine-Tuning",
  5: "Save Model",
};

export default function CreateModel({
  selectedTrainingRun,
}: {
  selectedTrainingRun: TrainingRunInterface;
}): ReactElement {
  const isMounted = useRef(false);
  const [current, setCurrent] = useState(0);
  const history = useHistory();
  const [trainingRun, setTrainingRun] = useState<TrainingRunInterface>(null);
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const selectedIsRunning = running?.ID === selectedTrainingRun.ID;

  const pollTrainingRun = () => {
    let stage: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"]["stage"] = null;
    if (current === 2) {
      stage = "acoustic";
    } else if (current === 4) {
      stage = "vocoder";
    }
    console.log(stage);
    const args: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      ID: selectedTrainingRun.ID,
      stage,
    };
    ipcRenderer
      .invoke(FETCH_TRAINING_RUNS_CHANNEL.IN, args)
      .then((trainingRuns: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        setTrainingRun(trainingRuns[0]);
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

  useInterval(() => {
    pollTrainingRun();
  }, POLL_LOGFILE_INTERVALL);

  /**
   * TODO Display training run name in Breadcrumb like it is done in preprocessing runs
   */
  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <BreadcrumbItem to={TRAINING_RUNS_ROUTE.RUN_SELECTION.ROUTE}>
          Training Runs
        </BreadcrumbItem>
        <BreadcrumbItem>{stepToTitle[current]}</BreadcrumbItem>
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
                title={getProgressTitle(
                  stepToTitle[1],
                  trainingRun === null
                    ? null
                    : (trainingRun.preprocessingCopyingFilesProgress +
                        trainingRun.preprocessingGenVocabProgress +
                        trainingRun.preprocessingGenAlignProgress +
                        trainingRun.preprocessingExtractDataProgress) /
                        4.0
                )}
                disabled={
                  trainingRun === null ||
                  ["not_started"].includes(trainingRun.stage)
                }
                icon={
                  selectedIsRunning &&
                  trainingRun?.stage === "preprocessing" ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                disabled={
                  trainingRun === null ||
                  ["not_started", "preprocessing"].includes(trainingRun.stage)
                }
                title={getProgressTitle(
                  stepToTitle[2],
                  trainingRun === null
                    ? null
                    : trainingRun.acousticFineTuningProgress
                )}
                icon={
                  selectedIsRunning &&
                  trainingRun?.stage === "acoustic_fine_tuning" ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                disabled={
                  trainingRun === null ||
                  [
                    "not_started",
                    "preprocessing",
                    "acoustic_fine_tuning",
                  ].includes(trainingRun.stage)
                }
                title={getProgressTitle(
                  stepToTitle[3],
                  trainingRun === null
                    ? null
                    : trainingRun.groundTruthAlignmentProgress
                )}
                icon={
                  selectedIsRunning &&
                  trainingRun?.stage === "ground_truth_alignment" ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                disabled={
                  trainingRun === null ||
                  [
                    "not_started",
                    "preprocessing",
                    "acoustic_fine_tuning",
                    "ground_truth_alignment",
                  ].includes(trainingRun.stage)
                }
                title={getProgressTitle(
                  stepToTitle[4],
                  trainingRun === null
                    ? null
                    : trainingRun.vocoderFineTuningProgress
                )}
                icon={
                  selectedIsRunning &&
                  trainingRun?.stage === "vocoder_fine_tuning" ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                disabled={
                  trainingRun === null ||
                  [
                    "not_started",
                    "preprocessing",
                    "acoustic_fine_tuning",
                    "ground_truth_alignment",
                    "vocoder_fine_tuning",
                  ].includes(trainingRun.stage)
                }
                title={stepToTitle[5]}
                icon={
                  selectedIsRunning && trainingRun?.stage === "save_model" ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
            </Steps>
          </Card>
        </Col>
        <Col span={20}>
          <Switch>
            <Route
              render={(props) =>
                trainingRun && (
                  <Configuration
                    onStepChange={onStepChange}
                    run={trainingRun}
                  ></Configuration>
                )
              }
              path={stepToPath[0]}
            ></Route>
            <Route
              render={(props) =>
                trainingRun && (
                  <Preprocessing
                    onStepChange={onStepChange}
                    run={trainingRun}
                  />
                )
              }
              path={stepToPath[1]}
            ></Route>
            <Route
              render={(props) =>
                trainingRun && (
                  <AcousticModelFinetuning
                    onStepChange={onStepChange}
                    run={trainingRun}
                  />
                )
              }
              path={stepToPath[2]}
            ></Route>
            <Route
              render={(props) =>
                trainingRun && (
                  <GroundTruthAlignment
                    onStepChange={onStepChange}
                    run={trainingRun}
                  />
                )
              }
              path={stepToPath[3]}
            ></Route>
            <Route
              render={(props) =>
                trainingRun && (
                  <VocoderFineTuning
                    onStepChange={onStepChange}
                    run={trainingRun}
                  />
                )
              }
              path={stepToPath[4]}
            ></Route>
            <Route
              render={(props) =>
                trainingRun && (
                  <SaveModel onStepChange={onStepChange} run={trainingRun} />
                )
              }
              path={stepToPath[5]}
            ></Route>
          </Switch>
        </Col>
      </Row>
    </>
  );
}
