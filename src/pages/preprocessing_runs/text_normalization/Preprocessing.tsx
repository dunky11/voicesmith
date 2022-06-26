import React, { ReactElement } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../../app/store";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import {
  RunInterface,
  TextNormalizationRunInterface,
} from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../../features/runManagerSlice";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
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
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(
        addToQueue({ ID: run.ID, type: "textNormalizationRun", name: run.name })
      );
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
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
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
