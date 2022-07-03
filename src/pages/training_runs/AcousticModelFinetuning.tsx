import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Tabs, Card, Button } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import AcousticStatistics from "./AcousticStatistics";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import RunCard from "../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../features/runManagerSlice";

export default function AcousticModelFinetuning({
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
  const isMounted = useRef(false);

  const stageIsRunning = getStageIsRunning(
    ["acoustic_fine_tuning"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["acoustic_fine_tuning"],
    run.stage,
    running,
    "trainingRun",
    run.ID
  );

  const onBackClick = () => {
    onStepChange(2);
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
      onStepChange(4);
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

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <RunCard
      title="Acoustic Model Fine-Tuning"
      docsUrl="/usage/training#acoustic-model-fine-tuning"
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button
          type="primary"
          disabled={
            run.stage === "not_started" || run.stage === "preprocessing"
          }
          onClick={onNextClick}
        >
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
          <AcousticStatistics
            audioStatistics={run.audioStatistics}
            imageStatistics={run.imageStatistics}
            graphStatistics={run.graphStatistics}
          ></AcousticStatistics>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(run.ID)}
            logFileName="acoustic_fine_tuning.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
