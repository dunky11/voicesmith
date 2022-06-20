import React, { ReactElement, useState } from "react";
import { Tabs, Card, Button, Steps } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import RunCard from "../../components/cards/RunCard";
import LogPrinter from "../../components/log_printer/LogPrinter";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import {
  RunInterface,
  TrainingRunInterface,
  UsageStatsInterface,
} from "../../interfaces";

export default function GroundTruthAlignment({
  onStepChange,
  trainingRun,
  running,
  continueRun,
  stopRun,
  usageStats,
}: {
  onStepChange: (step: number) => void;
  trainingRun: TrainingRunInterface;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;

  usageStats: UsageStatsInterface[];
}): ReactElement {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");

  const stageIsRunning = getStageIsRunning(
    ["ground_truth_alignment"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["ground_truth_alignment"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const onBackClick = () => {
    onStepChange(3);
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
    } else {
      onStepChange(5);
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

  return (
    <RunCard
      buttons={[
        <Button style={{ marginRight: 8 }} onClick={onBackClick}>
          Back
        </Button>,
        <Button type="primary" onClick={onNextClick}>
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
            <Steps direction="vertical" size="small" current={0}>
              <Steps.Step
                title="Generate ground truth alignments"
                description="Save predictions of the acoustic model on disk to fine-tune the vocoder."
                icon={stageIsRunning ? <LoadingOutlined /> : undefined}
              />
            </Steps>
          </Card>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(trainingRun.ID)}
            logFileName="ground_truth_alignment.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
