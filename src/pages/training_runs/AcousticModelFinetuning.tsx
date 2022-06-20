import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Tabs, Card, Button } from "antd";
import AcousticStatistics from "./AcousticStatistics";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../components/log_printer/LogPrinter";
import {
  AudioStatisticInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
  RunInterface,
  TrainingRunInterface,
  UsageStatsInterface,
} from "../../interfaces";
import {
  useInterval,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import RunCard from "../../components/cards/RunCard";
import {
  FETCH_TRAINING_RUNS_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL_TYPES,
} from "../../channels";
const { ipcRenderer } = window.require("electron");

export default function AcousticModelFinetuning({
  onStepChange,
  running,
  continueRun,
  stopRun,
  trainingRun,
  usageStats,
}: {
  onStepChange: (step: number) => void;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
  trainingRun: TrainingRunInterface;
  usageStats: UsageStatsInterface[];
}): ReactElement {
  const [selectedTab, setSelectedTab] = useState<string>("Overview");
  const [graphStatistics, setGraphStatistics] = useState<
    GraphStatisticInterface[]
  >([]);
  const [imageStatistics, setImageStatistics] = useState<
    ImageStatisticInterface[]
  >([]);
  const [audioStatistics, setAudioStatistics] = useState<
    AudioStatisticInterface[]
  >([]);
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

  const pollStatistics = () => {
    const args: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      withStatistics: false,
      stage: "acoustic",
      ID: trainingRun.ID,
    };
    ipcRenderer
      .invoke(FETCH_TRAINING_RUNS_CHANNEL.IN, args)
      .then((runs: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        setGraphStatistics(runs[0].graphStatistics);
        setImageStatistics(runs[0].imageStatistics);
        setAudioStatistics(runs[0].audioStatistics);
      });
  };

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

  useInterval(pollStatistics, 5000);

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
      <Tabs defaultActiveKey="Overview" onChange={setSelectedTab}>
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow
            usageStats={usageStats}
            style={{ marginBottom: 16 }}
          ></UsageStatsRow>
          <AcousticStatistics
            audioStatistics={audioStatistics}
            imageStatistics={imageStatistics}
            graphStatistics={graphStatistics}
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
