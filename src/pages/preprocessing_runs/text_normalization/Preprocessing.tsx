import React, { ReactElement } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import {
  RunInterface,
  TextNormalizationRunInterface,
  UsageStatsInterface,
} from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";

export default function Preprocessing({
  onStepChange,
  run,
  running,
  continueRun,
  usageStats,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  usageStats: UsageStatsInterface[];
  stopRun: () => void;
}): ReactElement {
  const stageIsRunning = getStageIsRunning(
    ["not_started", "text_normalization"],
    run.stage,
    running,
    "textNormalizationRun",
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["not_started", "text_normalization"],
    run.stage,
    running,
    "textNormalizationRun",
    run.ID
  );

  const onBackClick = () => {
    onStepChange(0);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({ ID: run.ID, type: "textNormalizationRun", name: run.name });
    } else if (["choose_samples", "finished"].includes(run.stage)) {
      onStepChange(2);
    }
  };

  const getNextButtonText = () => {
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
        <Button type="primary" onClick={onNextClick}>
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview">
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
                  run.textNormalizationProgress
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
            name={String(run.ID)}
            logFileName="preprocessing.txt"
            type="textNormalizationRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
