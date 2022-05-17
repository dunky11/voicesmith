import React, { useEffect, useState, useRef } from "react";
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
  selectedTrainingRunID: number | null;
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
}) {
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
        "fetch-training-run-statistics",
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
    if (selectedTrainingRunID === null) {
      return;
    }
    if (wouldContinueRun) {
      continueRun({ ID: selectedTrainingRunID, type: "trainingRun" });
    } else if (stageIsRunning) {
      stopRun();
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
    <Card
      actions={[
        <div
          key="next-button-wrapper"
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginRight: 24,
          }}
        >
          <Button style={{ marginRight: 8 }} onClick={onBackClick}>
            Back
          </Button>
          <Button
            type="primary"
            disabled={
              selectedTrainingRunID === null ||
              stage === "not_started" ||
              stage === "preprocessing"
            }
            onClick={onNextClick}
          >
            {getNextButtonText()}
          </Button>
        </div>,
      ]}
      bodyStyle={{ paddingTop: 8 }}
    >
      <Tabs defaultActiveKey="Overview" onChange={setSelectedTab}>
        <Tabs.TabPane tab="Overview" key="overview">
          <UsageStatsRow
            usageStats={usageStats}
            style={{ marginBottom: selectedTrainingRunID === null ? 0 : 16 }}
          ></UsageStatsRow>
          {selectedTrainingRunID !== null && (
            <AcousticStatistics
              audioStatistics={audioStatistics}
              imageStatistics={imageStatistics}
              graphStatistics={graphStatistics}
            ></AcousticStatistics>
          )}
        </Tabs.TabPane>
        <Tabs.TabPane tab="Log" key="log">
          <LogPrinter
            name={
              selectedTrainingRunID === null
                ? null
                : String(selectedTrainingRunID)
            }
            logFileName="acoustic_fine_tuning.txt"
            type="trainingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </Card>
  );
}
