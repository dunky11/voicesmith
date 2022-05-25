import React, { useState, useEffect, useRef } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
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
}) {
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
      ipcRenderer.invoke("remove-training-run", ID);
    }
  };

  useEffect(() => {
    if (removeTrainingRunID.current !== null) {
      ipcRenderer.invoke("remove-training-run", removeTrainingRunID.current);
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
