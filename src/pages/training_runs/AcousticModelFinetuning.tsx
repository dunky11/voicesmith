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
  UsageStatsInterface,
} from "../../interfaces";
import { POLL_LOGFILE_INTERVALL } from "../../config";
import {
  useInterval,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import RunCard from "../../components/cards/RunCard";
import { FETCH_TRAINING_RUN_STATISTICS_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");

export default function AcousticModelFinetuning({
  onStepChange,
  selectedTrainingRunID,
  running,
  continueRun,
  stopRun,
  stage,
  usageStats,
}: {
  onStepChange: (step: number) => void;
  selectedTrainingRunID: number;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
  stage:
    | "not_started"
    | "preprocessing"
    | "acoustic_fine_tuning"
    | "ground_truth_alignment"
    | "vocoder_fine_tuning"
    | "save_model"
    | "finished"
    | null;
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
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );
  const wouldContinueRun = getWouldContinueRun(
    ["acoustic_fine_tuning"],
    stage,
    running,
    "trainingRun",
    selectedTrainingRunID
  );

  const pollStatistics = () => {
    ipcRenderer
      .invoke(
        FETCH_TRAINING_RUN_STATISTICS_CHANNEL.IN,
        selectedTrainingRunID,
        "acoustic"
      )
      .then(
        (statistics: {
          graphStatistics: GraphStatisticInterface[];
          imageStatistics: ImageStatisticInterface[];
          audioStatistics: AudioStatisticInterface[];
        }) => {
          if (!isMounted.current) {
            return;
          }
          setGraphStatistics(statistics.graphStatistics);
          setImageStatistics(statistics.imageStatistics);
          setAudioStatistics(statistics.audioStatistics);
        }
      );
  };

  const onBackClick = () => {
    onStepChange(2);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      stopRun();
    } else if (wouldContinueRun) {
      continueRun({ ID: selectedTrainingRunID, type: "trainingRun" });
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
          disabled={stage === "not_started" || stage === "preprocessing"}
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
            name={String(selectedTrainingRunID)}
            logFileName="acoustic_fine_tuning.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
