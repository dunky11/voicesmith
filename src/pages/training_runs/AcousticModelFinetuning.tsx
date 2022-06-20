import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Tabs, Card, Button } from "antd";
import AcousticStatistics from "./AcousticStatistics";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import RunCard from "../../components/cards/RunCard";

export default function AcousticModelFinetuning({
  onStepChange,
  running,
  continueRun,
  stopRun,
  trainingRun,
}: {
  onStepChange: (step: number) => void;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
  trainingRun: TrainingRunInterface;
}): ReactElement {
  const isMounted = useRef(false);

  const stageIsRunning = getStageIsRunning(
    ["acoustic_fine_tuning"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["acoustic_fine_tuning"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const onBackClick = () => {
    onStepChange(2);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({
        ID: trainingRun.ID,
        type: "trainingRun",
        name: trainingRun.name,
      });
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
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button
          type="primary"
          disabled={
            trainingRun.stage === "not_started" ||
            trainingRun.stage === "preprocessing"
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
            audioStatistics={trainingRun.audioStatistics}
            imageStatistics={trainingRun.imageStatistics}
            graphStatistics={trainingRun.graphStatistics}
          ></AcousticStatistics>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(trainingRun.ID)}
            logFileName="acoustic_fine_tuning.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
