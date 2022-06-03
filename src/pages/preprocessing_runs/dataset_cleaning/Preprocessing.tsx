import React, { useState } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import { RunInterface, UsageStatsInterface } from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";

export default function Configuration({
  onStepChange,
  selectedID,
  running,
  continueRun,
  stage,
  usageStats,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  selectedID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stage:
    | "not_started"
    | "gen_file_embeddings"
    | "detect_outliers"
    | "choose_samples"
    | "apply_changes"
    | "finished"
    | null;
  usageStats: UsageStatsInterface[];
  stopRun: () => void;
}) {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");

  const stageIsRunning = getStageIsRunning(
    ["not_started", "gen_file_embeddings", "detect_outliers"],
    stage,
    running,
    "dSCleaning",
    selectedID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["not_started", "gen_file_embeddings", "detect_outliers"],
    stage,
    running,
    "dSCleaning",
    selectedID
  );

  const onBackClick = () => {
    onStepChange(0);
  };

  const onNextClick = () => {
    if (selectedID === null) {
      return;
    }
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({ ID: selectedID, type: "dSCleaningRun" });
    } else if (
      ["choose_samples", "apply_changes", "finished"].includes(stage)
    ) {
      onStepChange(2);
    }
  };

  const getNextButtonText = () => {
    if (selectedID === null) {
      return "Next";
    }
    if (stageIsRunning) {
      return "Pause Run";
    }
    if (wouldContinueRun) {
      return "Continue Run";
    }
    return "Next";
  };

  const getCurrent = () => {
    if (selectedID === null || stage == null) {
      return 0;
    }
    switch (stage) {
      case "not_started":
        return 0;
      case "gen_file_embeddings":
        return 0;
      case "detect_outliers":
        return 1;
      case "choose_samples":
        return 1;
      case "apply_changes":
        return 1;
      case "finished":
        return 1;
      default:
        throw new Error(
          `No case selected in switch-statemen, '${stage}' is not a valid case ...`
        );
    }
  };

  const current = getCurrent();

  return (
    <RunCard
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button
          type="primary"
          disabled={selectedID === null}
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
                title="Generating File Embeddings"
                description="Generating an embedding vector for each file."
                icon={
                  current === 0 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title="Detecting Outliers"
                description="Detecting outliers in the dataset."
                icon={
                  current === 1 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
            </Steps>
          </Card>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={selectedID === null ? null : String(selectedID)}
            logFileName="preprocessing.txt"
            type="cleaningRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
