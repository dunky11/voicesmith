import React, { ReactElement, useState } from "react";
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

export default function Preprocessing({
  onStepChange,
  selectedID,
  running,
  continueRun,
  stage,
  textNormalizationProgress,
  usageStats,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  selectedID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stage:
    | "not_started"
    | "text_normalization"
    | "choose_samples"
    | "finished"
    | null;
  textNormalizationProgress: number;
  usageStats: UsageStatsInterface[];
  stopRun: () => void;
}): ReactElement {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");

  const stageIsRunning = getStageIsRunning(
    ["not_started", "text_normalization"],
    stage,
    running,
    "textNormalizationRun",
    selectedID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["not_started", "text_normalization"],
    stage,
    running,
    "textNormalizationRun",
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
      continueRun({ ID: selectedID, type: "textNormalizationRun" });
    } else if (["choose_samples", "finished"].includes(stage)) {
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

  const current = 0;

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
                title={getProgressTitle(
                  "Text Normalization",
                  textNormalizationProgress
                )}
                description="Normalizing text of each file."
                icon={
                  current === 0 && stageIsRunning ? (
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
            type="textNormalizationRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
