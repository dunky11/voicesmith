import React, { ReactElement, useState } from "react";
import { Tabs, Steps, Button, Card } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../../app/store";
import UsageStatsRow from "../../../components/usage_stats/UsageStatsRow";
import LogPrinter from "../../../components/log_printer/LogPrinter";
import { RunInterface, SampleSplittingRunInterface } from "../../../interfaces";
import { LoadingOutlined } from "@ant-design/icons";
import {
  getProgressTitle,
  getStageIsRunning,
  getWouldContinueRun,
} from "../../../utils";
import RunCard from "../../../components/cards/RunCard";
import { setIsRunning, addToQueue } from "../../../features/runManagerSlice";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const stageIsRunning = getStageIsRunning(
    [
      "not_started",
      "copying_files",
      "gen_vocab",
      "gen_alignments",
      "creating_splits",
    ],
    run.stage,
    running,
    "sampleSplittingRun",
    run.ID
  );

  const wouldContinueRun = getWouldContinueRun(
    [
      "not_started",
      "copying_files",
      "gen_vocab",
      "gen_alignments",
      "creating_splits",
    ],
    run.stage,
    running,
    "sampleSplittingRun",
    run.ID
  );

  const onBackClick = () => {
    onStepChange(0);
  };

  const onNextClick = () => {
    if (stageIsRunning) {
      dispatch(setIsRunning(false));
    } else if (wouldContinueRun) {
      dispatch(
        addToQueue({ ID: run.ID, type: "sampleSplittingRun", name: run.name })
      );
    } else if (["choose_samples", "finished"].includes(run.stage)) {
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
      case "copying_files":
        return 0;
      case "gen_vocab":
        return 1;
      case "gen_alignments":
        return 2;
      case "creating_splits":
        return 3;
      case "choose_samples":
        return 3;
      case "apply_changes":
        return 3;
      case "finished":
        return 3;
      default:
        throw new Error(
          `No case selected in switch-statemen, '${run.stage}' is not a valid case ...`
        );
    }
  };

  const current = getCurrent();

  return (
    <RunCard
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button type="primary" disabled={run === null} onClick={onNextClick}>
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
                title={getProgressTitle(
                  "Copy Files",
                  run ? run.copyingFilesProgress : 0
                )}
                description="Copy text and audio files into the correct folder."
                icon={
                  current === 0 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={getProgressTitle(
                  "Generate Vocabulary",
                  run ? run.genVocabProgress : 0
                )}
                description="Generate phonemes for each word."
                icon={
                  current === 1 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={getProgressTitle(
                  "Generate Alignments",
                  run ? run.genAlignProgress : 0
                )}
                description="Generate timestamps for each phoneme."
                icon={
                  current === 2 && stageIsRunning ? (
                    <LoadingOutlined />
                  ) : undefined
                }
              />
              <Steps.Step
                title={getProgressTitle(
                  "Creating Splits",
                  run.creatingSplitsProgress
                )}
                description="Splitting audios by sentence boundaries."
                icon={
                  current === 3 && stageIsRunning ? (
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
            type="sampleSplittingRun"
          />
        </Tabs.TabPane>
      </Tabs>
    </RunCard>
  );
}
