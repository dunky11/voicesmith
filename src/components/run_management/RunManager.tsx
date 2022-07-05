import React, { useEffect, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import { notification } from "antd";
import {
  CONTINUE_CLEANING_RUN_CHANNEL,
  CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL,
  CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL,
  CONTINUE_TRAINING_RUN_CHANNEL,
  STOP_RUN_CHANNEL,
} from "../../channels";
import { setIsRunning, popFromQueue } from "../../features/runManagerSlice";
import { RunInterface } from "../../interfaces";
import { RootState } from "../../app/store";

const { ipcRenderer } = window.require("electron");

export default function RunManager(): React.ReactElement {
  const runManager = useSelector((state: RootState) => state.runManager);
  const dispatch = useDispatch();
  const onRunFinishRef = useRef<() => void>(null);

  const onRunFinish = () => {
    if (runManager.queue.length === 0) {
      return;
    }
    if (runManager.queue.length > 1) {
      continueRun(runManager.queue[1]);
    } else {
      dispatch(setIsRunning(false));
    }
    dispatch(popFromQueue());
  };
  onRunFinishRef.current = onRunFinish;

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
            onRunFinishRef.current();
            return;
          case "error":
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

  const stopRun = () => {
    ipcRenderer.invoke(STOP_RUN_CHANNEL.IN);
  };

  useEffect(() => {
    if (runManager.isRunning) {
      if (runManager.queue.length === 0) {
        throw Error("Cant start run when queue is empty ...");
      }
      continueRun(runManager.queue[0]);
    } else {
      stopRun();
    }
  }, [runManager.isRunning]);

  return null;
}
