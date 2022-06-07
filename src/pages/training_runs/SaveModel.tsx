import React, { ReactElement } from "react";
import { Tabs, Card, Button, Steps } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import RunCard from "../../components/cards/RunCard";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, UsageStatsInterface } from "../../interfaces";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";

export default function VocoderFineTuning({
  onStepChange,
  selectedTrainingRunID,
  running,
  continueRun,
  stopRun,
  stage,
  usageStats,
}: {
  onStepChange: (step: number) => void;
  selectedTrainingRunID: number;
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
  usageStats: UsageStatsInterface[];
}): ReactElement {
  const stageIsRunning = getStageIsRunning(
    ["save_model"],
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["save_model"],
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );

  const onBackClick = () => {
    onStepChange(5);
  };

  const onNextClick = () => {
    if (wouldContinueRun) {
      continueRun({ ID: selectedTrainingRunID, type: "trainingRun" });
    } else if (stageIsRunning) {
      stopRun();
    }
  };

  const getNextButtonText = () => {
    if (stageIsRunning) {
      return "Pause Training";
    }
    if (wouldContinueRun) {
      return "Continue Training";
    }
  };

  return (
    <RunCard
      buttons={[
        <Button style={{ marginRight: 8 }} onClick={onBackClick}>
          Back
        </Button>,
        (stageIsRunning || wouldContinueRun) && (
          <Button type="primary" onClick={onNextClick}>
            {getNextButtonText()}
          </Button>
        ),
      ]}
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow usageStats={usageStats}></UsageStatsRow>
          <Card title="Progress">
            <Steps direction="vertical" size="small" current={0}>
              <Steps.Step
                title="Save model"
                icon={stageIsRunning ? <LoadingOutlined /> : undefined}
              />
            </Steps>
          </Card>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(selectedTrainingRunID)}
            logFileName="save_model.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
