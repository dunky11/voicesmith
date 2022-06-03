import React, { ReactElement, useState } from "react";
import { Tabs, Card, Steps, Button } from "antd";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, UsageStatsInterface } from "../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import RunCard from "../../components/cards/RunCard";

export default function Preprocessing({
  onStepChange,
  selectedTrainingRunID,
  running,
  continueRun,
  stopRun,
  stage,
  preprocessingStage,
  usageStats,
  copyingFilesProgress,
  genVocabProgress,
  genAlignProgress,
  extractDataProgress,
}: {
  onStepChange: (step: number) => void;
  selectedTrainingRunID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
  stage:
    | "not_started"
    | "preprocessing"
    | "acoustic_fine_tuning"
    | "ground_truth_alignment"
    | "vocoder_fine_tuning"
    | "save_model"
    | "finished"
    | null;
  preprocessingStage:
    | "not_started"
    | "copying_files"
    | "gen_vocab"
    | "gen_alignments"
    | "extract_data"
    | "finished"
    | null;
  usageStats: UsageStatsInterface[];
  copyingFilesProgress: number | null;
  genVocabProgress: number | null;
  genAlignProgress: number | null;
  extractDataProgress: number | null;
}): ReactElement {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");

  const stageIsRunning = getStageIsRunning(
    ["preprocessing"],
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["preprocessing"],
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );

  const getCurrent = () => {
    if (selectedTrainingRunID === null || preprocessingStage == null) {
      return 0;
    }
    switch (preprocessingStage) {
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
          `No case selected in switch-statemen, '${preprocessingStage}' is not a valid case ...`
        );
    }
  };

  const onBackClick = () => {
    onStepChange(1);
  };

  const onNextClick = () => {
    if (selectedTrainingRunID === null) {
      return;
    }
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({ ID: selectedTrainingRunID, type: "trainingRun" });
    } else if (stage === "finished") {
      onStepChange(3);
    }
  };

  const getNextButtonText = () => {
    if (selectedTrainingRunID === null) {
      return "Next";
    }
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
        <Button
          type="primary"
          disabled={selectedTrainingRunID === null}
          onClick={onNextClick}
        >
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview" onChange={setSelectedTab}>
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow
            usageStats={usageStats}
            style={{ marginBottom: 16 }}
          ></UsageStatsRow>
          <Card title="Progress">
            <Steps direction="vertical" size="small" current={current}>
              <Steps.Step
                title={getProgressTitle("Copy Files", copyingFilesProgress)}
                description="Copy text and audio files into the training folder."
                icon={
                  current === 0 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={
                  genVocabProgress === 0
                    ? "Generate Vocabulary"
                    : getProgressTitle("Generate Vocabulary", genVocabProgress)
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
                  genAlignProgress === 0
                    ? "Generate Alignments"
                    : getProgressTitle("Generate Alignments", genAlignProgress)
                }
                description="Generate timestamps for each phoneme."
                icon={
                  current === 2 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={getProgressTitle("Extract Data", extractDataProgress)}
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
            name={
              selectedTrainingRunID === null
                ? null
                : String(selectedTrainingRunID)
            }
            logFileName="preprocessing.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
