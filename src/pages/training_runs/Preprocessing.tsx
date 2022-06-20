import React, { ReactElement } from "react";
import { Tabs, Card, Steps, Button } from "antd";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import {
  RunInterface,
  TrainingRunInterface,
  UsageStatsInterface,
} from "../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import RunCard from "../../components/cards/RunCard";

export default function Preprocessing({
  onStepChange,
  trainingRun,
  running,
  continueRun,
  stopRun,
}: {
  onStepChange: (step: number) => void;
  trainingRun: TrainingRunInterface;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
}): ReactElement {
  const stageIsRunning = getStageIsRunning(
    ["preprocessing"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["preprocessing"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const getCurrent = () => {
    switch (trainingRun.preprocessingStage) {
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
          `No case selected in switch-statemen, '${trainingRun.preprocessingStage}' is not a valid case ...`
        );
    }
  };

  const onBackClick = () => {
    onStepChange(1);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({
        ID: trainingRun.ID,
        type: "trainingRun",
        name: trainingRun.name,
      });
    } else if (trainingRun.stage === "finished") {
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
                  trainingRun.preprocessingCopyingFilesProgress
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
                  trainingRun.preprocessingGenVocabProgress === 0
                    ? "Generate Vocabulary"
                    : getProgressTitle(
                        "Generate Vocabulary",
                        trainingRun.preprocessingGenVocabProgress
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
                  trainingRun.preprocessingGenAlignProgress === 0
                    ? "Generate Alignments"
                    : getProgressTitle(
                        "Generate Alignments",
                        trainingRun.preprocessingGenAlignProgress
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
                  trainingRun.preprocessingExtractDataProgress
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
            name={String(trainingRun.ID)}
            logFileName="preprocessing.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
