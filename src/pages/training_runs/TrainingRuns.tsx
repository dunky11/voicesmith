import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import { TRAINING_RUNS_ROUTE } from "../../routes";
import { REMOVE_TRAINING_RUN_CHANNEL } from "../../channels";
import { TrainingRunInterface } from "../../interfaces";
import CreateModel from "./CreateModel";
import RunSelection from "./RunSelection";
const { ipcRenderer } = window.require("electron");

export default function TrainingRuns(): ReactElement {
  const history = useHistory();
  const trainingRunToRm = useRef<TrainingRunInterface | null>(null);
  const [selectedTrainingRun, setSelectedTrainingRun] =
    useState<TrainingRunInterface | null>(null);

  const selectTrainingRun = (run: TrainingRunInterface) => {
    trainingRunToRm.current = null;
    setSelectedTrainingRun(run);
    history.push(TRAINING_RUNS_ROUTE.CREATE_MODEL.ROUTE);
  };

  const removeTrainingRun = (run: TrainingRunInterface) => {
    if (selectedTrainingRun !== null && run.ID === selectedTrainingRun.ID) {
      trainingRunToRm.current = run;
      setSelectedTrainingRun(null);
    } else {
      ipcRenderer.invoke(REMOVE_TRAINING_RUN_CHANNEL.IN, run.ID);
    }
  };

  useEffect(() => {
    if (trainingRunToRm.current !== null) {
      ipcRenderer.invoke(
        REMOVE_TRAINING_RUN_CHANNEL.IN,
        trainingRunToRm.current.ID
      );
      trainingRunToRm.current = null;
    }
  }, [selectedTrainingRun]);

  return (
    <Switch>
      <Route
        render={() =>
          selectedTrainingRun && (
            <CreateModel selectedTrainingRun={selectedTrainingRun} />
          )
        }
        path={TRAINING_RUNS_ROUTE.CREATE_MODEL.ROUTE}
      ></Route>
      <Route
        render={() => (
          <RunSelection
            selectTrainingRun={selectTrainingRun}
            removeTrainingRun={removeTrainingRun}
          />
        )}
        path={TRAINING_RUNS_ROUTE.RUN_SELECTION.ROUTE}
      ></Route>
    </Switch>
  );
}
