import React, { ReactElement } from "react";
import { Tabs, Card, Button, Steps } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import RunCard from "../../components/cards/RunCard";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";

export default function SaveModel({
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
    ["save_model"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["save_model"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const onBackClick = () => {
    onStepChange(5);
  };

  const onNextClick = () => {
    if (wouldContinueRun) {
      continueRun({
        ID: trainingRun.ID,
        type: "trainingRun",
        name: trainingRun.name,
      });
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
          <UsageStatsRow></UsageStatsRow>
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
            name={String(trainingRun.ID)}
            logFileName="save_model.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
