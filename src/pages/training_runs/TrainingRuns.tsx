import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Route, Switch, useHistory } from "react-router-dom";
import { TRAINING_RUNS_ROUTE } from "../../routes";
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
    setSelectedTrainingRunID(ID);
    history.push(TRAINING_RUNS_ROUTE.CREATE_MODEL.ROUTE);
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
    }
  }, [selectedTrainingRunID]);

  return (
    <Switch>
      <Route
        render={() =>
          selectedTrainingRunID && (
            <CreateModel
              selectedTrainingRunID={selectedTrainingRunID}
              setSelectedTrainingRunID={setSelectedTrainingRunID}
              running={running}
              continueRun={continueRun}
              stopRun={stopRun}
            />
          )
        }
        path={TRAINING_RUNS_ROUTE.CREATE_MODEL.ROUTE}
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
        path={TRAINING_RUNS_ROUTE.RUN_SELECTION.ROUTE}
      ></Route>
    </Switch>
  );
}
