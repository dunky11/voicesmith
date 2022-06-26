import React, { ReactElement } from "react";
import { Tabs, Card, Steps, Button } from "antd";
import { useSelector, useDispatch } from "react-redux";
import { LoadingOutlined } from "@ant-design/icons";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { RootState } from "../../app/store";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import RunCard from "../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../features/runManagerSlice";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (step: number) => void;
  run: TrainingRunInterface;
}): ReactElement {
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const runManager = useSelector((state: RootState) => state.runManager);
  const dispatch = useDispatch();
  const stageIsRunning = getStageIsRunning(
    ["preprocessing"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["preprocessing"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );

  const getCurrent = () => {
    switch (run.preprocessingStage) {
      case "not_started":
        return 0;
      case "copying_files":
        return 0;
      case "gen_vocab":
        return 1;
      case "gen_alignments":
        return 2;
      case "extract_data":
        return 3;
      case "finished":
        return 4;
      default:
        throw new Error(
          `No case selected in switch-statemen, '${run.preprocessingStage}' is not a valid case ...`
        );
    }
  };

  const onBackClick = () => {
    onStepChange(1);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(
        addToQueue({
          ID: run.ID,
          type: "trainingRun",
          name: run.name,
        })
      );
    } else if (run.stage === "finished") {
      onStepChange(3);
    }
  };

  const getNextButtonText = () => {
    if (stageIsRunning) {
      return "Pause Training";
    }
    if (wouldContinueRun) {
      return "Continue Training";
    }
    return "Next";
  };

  const current: number = getCurrent();

  return (
    <RunCard
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button type="primary" onClick={onNextClick}>
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
          <Card title="Progress">
            <Steps direction="vertical" size="small" current={current}>
              <Steps.Step
                title={getProgressTitle(
                  "Copy Files",
                  run.preprocessingCopyingFilesProgress
                )}
                description="Copy text and audio files into the training folder."
                icon={
                  current === 0 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={
                  run.preprocessingGenVocabProgress === 0
                    ? "Generate Vocabulary"
                    : getProgressTitle(
                        "Generate Vocabulary",
                        run.preprocessingGenVocabProgress
                      )
                }
                description="Generate phonemes for each word."
                icon={
                  current === 1 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={
                  run.preprocessingGenAlignProgress === 0
                    ? "Generate Alignments"
                    : getProgressTitle(
                        "Generate Alignments",
                        run.preprocessingGenAlignProgress
                      )
                }
                description="Generate timestamps for each phoneme."
                icon={
                  current === 2 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={getProgressTitle(
                  "Extract Data",
                  run.preprocessingExtractDataProgress
                )}
                description="Resample audio and extract pitch information and mel-spectrograms."
                icon={
                  current === 3 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
            </Steps>
          </Card>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(run.ID)}
            logFileName="preprocessing.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
