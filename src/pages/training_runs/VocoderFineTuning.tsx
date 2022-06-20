import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Tabs, Card, Button } from "antd";
import VocoderStatistics from "./VocoderStatistics";
import LogPrinter from "../../components/log_printer/LogPrinter";
import {
  GraphStatisticInterface,
  ImageStatisticInterface,
  AudioStatisticInterface,
  UsageStatsInterface,
  RunInterface,
  TrainingRunInterface,
} from "../../interfaces";
import {
  useInterval,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../utils";
import UsageStatsRow from "../../components/usage_stats/UsageStatsRow";
import RunCard from "../../components/cards/RunCard";
const { ipcRenderer } = window.require("electron");

export default function VocoderFineTuning({
  onStepChange,
  trainingRun,
  running,
  continueRun,
  stopRun,
  usageStats,
}: {
  onStepChange: (step: number) => void;
  trainingRun: TrainingRunInterface;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
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

  const pollStatistics = () => {
    ipcRenderer
      .invoke("fetch-training-run-statistics", trainingRun.ID, "vocoder")
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
        <Button type="primary" onClick={onNextClick}>
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
          <VocoderStatistics
            audioStatistics={audioStatistics}
            imageStatistics={imageStatistics}
            graphStatistics={graphStatistics}
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
