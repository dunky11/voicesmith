import React, { ReactElement } from "react";
import { Tabs, Card, Button, Steps } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import RunCard from "../../components/cards/RunCard";
import LogPrinter from "../../components/log_printer/LogPrinter";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { setIsRunning, addToQueue } from "../../features/runManagerSlice";

export default function GroundTruthAlignment({
  onStepChange,
  run,
}: {
  onStepChange: (step: number) => void;
  run: TrainingRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const stageIsRunning = getStageIsRunning(
    ["ground_truth_alignment"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["ground_truth_alignment"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );

  const onBackClick = () => {
    onStepChange(3);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(
        addToQueue({
          ID: run.ID,
          type: "trainingRun",
          name: run.name,
        })
      );
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
      title="Generate Ground Truth Alignments"
      docsUrl="/usage/training#generate-ground-truth-alignments"
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
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
            name={String(run.ID)}
            logFileName="ground_truth_alignment.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
