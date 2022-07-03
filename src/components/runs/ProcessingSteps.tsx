import React, { ReactElement } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import UsageStatsRow from "../usage_stats/UsageStatsRow";
import LogPrinter from "../log_printer/LogPrinter";
import { RunInterface } from "../../interfaces";
import {
  getStageIsRunning,
  getWouldContinueRun,
  getProgressTitle,
} from "../../utils";
import RunCard from "../cards/RunCard";
import { setIsRunning, addToQueue } from "../../features/runManagerSlice";

interface Stage {
  title: string;
  stageName: string;
  progress: number | null;
  description: string;
}

export default function ProcessingSteps({
  title,
  docsUrl,
  onBack,
  onNext,
  stage,
  run,
  logFileName,
  previousStages,
  currentStages,
  nextStages,
}: {
  title: string;
  docsUrl: string | null;
  onBack: () => void;
  onNext: (() => void) | null;
  stage: string;
  run: RunInterface;
  logFileName: string;
  previousStages: string[];
  currentStages: Stage[];
  nextStages: string[];
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const stageIsRunning = getStageIsRunning(
    [...previousStages, ...currentStages.map((stage) => stage.stageName)],
    stage,
    running,
    run.type,
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    [...previousStages, ...currentStages.map((stage) => stage.stageName)],
    stage,
    running,
    run.type,
    run.ID
  );

  const onNextClick = () => {
    if (stageIsRunning) {
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(addToQueue(run));
    } else if (nextStages.includes(stage) && onNext !== null) {
      onNext();
    }
  };

  const renderNextButton = (): ReactElement => {
    if (stageIsRunning) {
      return (
        <Button type="primary" onClick={onNextClick}>
          Pause Run
        </Button>
      );
    }
    if (wouldContinueRun) {
      return (
        <Button type="primary" onClick={onNextClick}>
          Continue Run
        </Button>
      );
    }
    if (onNext !== null) {
      return (
        <Button type="primary" onClick={onNextClick}>
          Next
        </Button>
      );
    }
    return null;
  };

  const getCurrent = () => {
    if (previousStages.includes(stage)) {
      return 0;
    }
    for (let i = 0; i < currentStages.length; i++) {
      if (currentStages[i].stageName === stage) {
        return i;
      }
    }
    if (nextStages.includes(stage)) {
      return currentStages.length - 1;
    }
    throw new Error(
      `Invalid stage received, '${stage}' is not a valid stage...`
    );
  };

  const current = getCurrent();

  return (
    <RunCard
      buttons={[<Button onClick={onBack}>Back</Button>, renderNextButton()]}
      title={title}
      docsUrl={docsUrl}
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
          <Card title="Progress">
            <Steps direction="vertical" size="small" current={current}>
              {currentStages.map((currentStage, index) => (
                <Steps.Step
                  title={getProgressTitle(
                    currentStage.title,
                    run ? currentStage.progress : 0
                  )}
                  description={currentStage.description}
                  icon={
                    current === index && stageIsRunning ? (
                      <LoadingOutlined />
                    ) : undefined
                  }
                  key={currentStage.stageName}
                />
              ))}
            </Steps>
          </Card>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(run.ID)}
            logFileName={logFileName}
            type={run.type}
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}

ProcessingSteps.defaultProps = {
  docsUrl: null,
};
