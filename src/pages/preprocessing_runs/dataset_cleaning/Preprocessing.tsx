import React, { ReactElement, useState } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import {
  CleaningRunInterface,
  RunInterface,
  UsageStatsInterface,
} from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import { getStageIsRunning, getWouldContinueRun } from "../../../utils";
import RunCard from "../../../components/cards/RunCard";

export default function Configuration({
  onStepChange,
  run,
  running,
  continueRun,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  run: CleaningRunInterface;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
}): ReactElement {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");

  const stageIsRunning = getStageIsRunning(
    ["not_started", "gen_file_embeddings", "detect_outliers"],
    run.stage,
    running,
    "cleaningRun",
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    ["not_started", "gen_file_embeddings", "detect_outliers"],
    run.stage,
    running,
    "cleaningRun",
    run.ID
  );

  const onBackClick = () => {
    onStepChange(0);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({ ID: run.ID, type: "cleaningRun", name: run.name });
    } else if (
      ["choose_samples", "apply_changes", "finished"].includes(run.stage)
    ) {
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

  const getCurrent = () => {
    switch (run.stage) {
      case "not_started":
        return 0;
      case "gen_file_embeddings":
        return 0;
      case "choose_samples":
        return 1;
      case "apply_changes":
        return 1;
      case "finished":
        return 1;
      default:
        throw new Error(
          `No case selected in switch-statement, '${run.stage}' is not a valid case ...`
        );
    }
  };

  const current = getCurrent();

  return (
    <RunCard
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button type="primary" onClick={onNextClick}>
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview" onChange={setSelectedTab}>
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
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
            name={String(run.ID)}
            logFileName="preprocessing.txt"
            type="cleaningRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
