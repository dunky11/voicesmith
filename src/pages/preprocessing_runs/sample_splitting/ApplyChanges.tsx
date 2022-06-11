import React, { ReactElement } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import {
  RunInterface,
  SampleSplittingRunInterface,
  UsageStatsInterface,
} from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";

export default function ApplyChanges({
  onStepChange,
  running,
  continueRun,
  run,
  usageStats,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  run: SampleSplittingRunInterface | null;
  usageStats: UsageStatsInterface[];
  stopRun: () => void;
}): ReactElement {
  const stageIsRunning = getStageIsRunning(
    ["apply_changes"],
    run.stage,
    running,
    "sampleSplittingRun",
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["apply_changes"],
    run.stage,
    running,
    "sampleSplittingRun",
    run.ID
  );

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (run.stage !== "finished") {
      continueRun({ ID: run.ID, type: "sampleSplittingRun" });
    }
  };

  const onBackClick = () => {
    onStepChange(2);
  };

  const getNextButtonText = () => {
    if (stageIsRunning) {
      return "Pause Run";
    }
    if (wouldContinueRun) {
      return "Continue Run";
    }
    return "";
  };

  const current = 0;

  // TODO progress for apply changes
  return (
    <RunCard
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        run.stage !== "finished" && (
          <Button type="primary" onClick={onNextClick}>
            {getNextButtonText()}
          </Button>
        ),
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
                  "Apply Changes",
                  run.applyingChangesProgress
                )}
                description="Splitting dataset by the sentence boundaries."
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
            name={run === null ? null : String(run.ID)}
            logFileName="apply_changes.txt"
            type="sampleSplittingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
