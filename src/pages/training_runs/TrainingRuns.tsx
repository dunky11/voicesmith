import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import { REMOVE_TRAINING_RUN_CHANNEL } from "../../channels";
import { RunInterface } from "../../interfaces";
import CreateModel from "./CreateModel";
import RunSelection from "./RunSelection";
const { ipcRenderer } = window.require("electron");

export default function TrainingRuns({
  running,
  continueRun,
  stopRun,
}: {
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
}): ReactElement {
  const history = useHistory();
  const removeTrainingRunID = useRef<number | null>();
  const [selectedTrainingRunID, setSelectedTrainingRunID] = useState<
    number | null
  >(null);

  const selectTrainingRun = (ID: number) => {
    removeTrainingRunID.current = null;
    if (selectedTrainingRunID == ID) {
      history.push("/training-runs/create-model");
    } else {
      setSelectedTrainingRunID(ID);
    }
  };

  const removeTrainingRun = (ID: number) => {
    if (ID === selectedTrainingRunID) {
      removeTrainingRunID.current = ID;
      setSelectedTrainingRunID(null);
    } else {
      ipcRenderer.invoke(REMOVE_TRAINING_RUN_CHANNEL.IN, ID);
    }
  };

  useEffect(() => {
    if (removeTrainingRunID.current !== null) {
      ipcRenderer.invoke(
        REMOVE_TRAINING_RUN_CHANNEL.IN,
        removeTrainingRunID.current
      );
      removeTrainingRunID.current = null;
    } else if (selectedTrainingRunID != null) {
      history.push("/training-runs/create-model");
    }
  }, [selectedTrainingRunID]);

  return (
    <Switch>
      <Route
        render={() => (
          <CreateModel
            selectedTrainingRunID={selectedTrainingRunID}
            setSelectedTrainingRunID={setSelectedTrainingRunID}
            running={running}
            continueRun={continueRun}
            stopRun={stopRun}
          />
        )}
        path="/training-runs/create-model"
      ></Route>
      <Route
        render={() => (
          <RunSelection
            selectTrainingRun={selectTrainingRun}
            removeTrainingRun={removeTrainingRun}
            running={running}
            stopRun={stopRun}
            continueRun={continueRun}
          />
        )}
        path="/training-runs/run-selection"
      ></Route>
    </Switch>
  );
}
