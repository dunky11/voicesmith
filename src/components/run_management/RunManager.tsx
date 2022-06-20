import React, { useEffect, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { notification } from "antd";
import {
  CONTINUE_CLEANING_RUN_CHANNEL,
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL,
  CONTINUE_TRAINING_RUN_CHANNEL,
} from "../../channels";
import { setIsRunning, popFromQueue } from "../../features/runManagerSlice";
import { RunInterface } from "../../interfaces";
import { RootState } from "../../app/store";

const { ipcRenderer } = window.require("electron");

export default function RunManager(): React.ReactElement {
  const runManager = useSelector((state: RootState) => state.runManager);
  const dispatch = useDispatch();
  const wasRunning = useRef(false);

  const onRunFinish = () => {
    if (runManager.queue.length === 0) {
      return;
    }
    if (runManager.queue.length > 1) {
      continueRun(runManager.queue[1]);
    } else {
      dispatch(setIsRunning(false));
    }
    dispatch(popFromQueue);
  };

  const continueRun = (run: RunInterface) => {
    ipcRenderer.removeAllListeners(CONTINUE_TRAINING_RUN_CHANNEL.REPLY);
    ipcRenderer.on(
      CONTINUE_TRAINING_RUN_CHANNEL.REPLY,
      (
        _: any,
        message: {
          type: string;
          errorMessage?: string;
        }
      ) => {
        switch (message.type) {
          case "notEnoughSpeakers": {
            notification["error"]({
              message: "Couldn't Start Run",
              description:
                "This runs dataset contains only one speaker, but it has to have at least two speakers in order to detect potentially noisy samples.",
              placement: "top",
            });
            return;
          }
          case "notEnoughSamples":
            notification["error"]({
              message: "Couldn't Start Run",
              description:
                "This runs dataset contains no samples. Please attach samples to the speakers in your dataset and try again.",
              placement: "top",
            });
            return;
          case "startedRun":
            return;
          case "finishedRun":
            onRunFinish();
            return;
          case "error":
            onRunFinish();
            notification["error"]({
              message: "Oops, an error occured, check logs for more info ...",
              description: message.errorMessage,
              placement: "top",
            });
            return;
          default:
            throw new Error(
              `No branch selected in switch-statement, '${message.type}' is not a valid case ...`
            );
        }
      }
    );
    switch (run.type) {
      case "trainingRun":
        ipcRenderer.send(CONTINUE_TRAINING_RUN_CHANNEL.IN, run.ID);
        break;
      case "cleaningRun":
        ipcRenderer.send(CONTINUE_CLEANING_RUN_CHANNEL.IN, run.ID);
        break;
      case "textNormalizationRun":
        ipcRenderer.send(CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL.IN, run.ID);
        break;
      case "sampleSplittingRun":
        ipcRenderer.send(CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL.IN, run.ID);
        break;
      default:
        throw new Error(
          `No branch selected in switch-statement, '${run.type}' is not a valid case ...`
        );
    }
  };

  useEffect(() => {
    if (runManager.isRunning && !wasRunning) {
      if (runManager.queue.length === 0) {
        throw Error("Cant start run when queue is empty ...");
      }
      continueRun(runManager.queue[0]);
    }
    wasRunning.current = runManager.isRunning;
  }, [runManager.isRunning]);

  return null;
}