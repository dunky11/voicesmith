import React, { ReactElement, useState } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../../app/store";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import {
  CleaningRunInterface,
  RunInterface,
  UsageStatsInterface,
} from "../../../interfaces";
import { getStageIsRunning, getWouldContinueRun } from "../../../utils";
import RunCard from "../../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../../features/runManagerSlice";

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: CleaningRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
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
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(addToQueue({ ID: run.ID, type: "cleaningRun", name: run.name }));
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
      <Tabs defaultActiveKey="Overview">
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
