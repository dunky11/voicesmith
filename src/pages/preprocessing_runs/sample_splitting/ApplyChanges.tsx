import React, { ReactElement } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../../app/store";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import { RunInterface, SampleSplittingRunInterface } from "../../../interfaces";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../../features/runManagerSlice";

export default function ApplyChanges({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface | null;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
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
      dispatch(setIsRunning(false));
    } else if (run.stage !== "finished") {
      dispatch(setIsRunning(true));
      dispatch(
        addToQueue({ ID: run.ID, type: "sampleSplittingRun", name: run.name })
      );
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
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
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
