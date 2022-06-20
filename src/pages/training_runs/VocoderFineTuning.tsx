import React, { useEffect, useRef, ReactElement } from "react";
import { Tabs, Button } from "antd";
import VocoderStatistics from "./VocoderStatistics";
import LogPrinter from "../../components/log_printer/LogPrinter";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { getStageIsRunning, getWouldContinueRun } from "../../utils";
import RunCard from "../../components/cards/RunCard";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";

export default function VocoderFineTuning({
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
  const isMounted = useRef(false);

  const stageIsRunning = getStageIsRunning(
    ["vocoder_fine_tuning"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["vocoder_fine_tuning"],
    trainingRun.stage,
    running,
    "trainingRun",
    trainingRun.ID
  );

  const onBackClick = () => {
    onStepChange(4);
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
      onStepChange(6);
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
        <Button type="primary" onClick={onNextClick}>
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Tabs defaultActiveKey="Overview">
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow style={{ marginBottom: 16 }}></UsageStatsRow>
          <VocoderStatistics
            audioStatistics={trainingRun.audioStatistics}
            imageStatistics={trainingRun.imageStatistics}
            graphStatistics={trainingRun.graphStatistics}
          ></VocoderStatistics>
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={String(trainingRun.ID)}
            logFileName="vocoder_fine_tuning.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
