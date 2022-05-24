import React, { useEffect, useState, useRef } from "react";
import { Switch, Route, useHistory } from "react-router-dom";
import { PreprocessingRunInterface, RunInterface } from "../../interfaces";
import PreprocessingRunSelection from "./PreprocessingRunSelection";
import TextNormalization from "./text_normalization/TextNormalization";
import DatasetCleaning from "./dataset_cleaning/DatasetCleaning";

export default function PreprcocessingRuns({
  running,
  continueRun,
  stopRun,
}: {
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stopRun: () => void;
}) {
  const isMounted = useRef(false);
  const history = useHistory();
  const [selectedPreprocessingRun, setSelectedPreprocessingRun] =
    useState<PreprocessingRunInterface | null>(null);

  useEffect(() => {
    if (selectedPreprocessingRun === null) {
      return;
    }
    switch (selectedPreprocessingRun.type) {
      case "textNormalizationRun":
        history.push("/preprocessing-runs/text-normalization/configuration");
        break;
      case "dSCleaningRun":
        history.push("/preprocessing-runs/dataset-cleaning/configuration");
        break;
      default:
        throw new Error(
          `No branch selected in switch-statement, case '${selectedPreprocessingRun.type}' is not a valid case`
        );
    }
  }, [selectedPreprocessingRun]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <Switch>
      <Route
        render={() => (
          <TextNormalization
            preprocessingRun={selectedPreprocessingRun}
            running={running}
            stopRun={stopRun}
            continueRun={continueRun}
          ></TextNormalization>
        )}
        path="/preprocessing-runs/text-normalization"
      ></Route>
      <Route
        path="/preprocessing-runs/dataset-cleaning"
        render={() => (
          <DatasetCleaning
            preprocessingRun={selectedPreprocessingRun}
            continueRun={continueRun}
            running={running}
            stopRun={stopRun}
          ></DatasetCleaning>
        )}
      ></Route>
      <Route
        render={() => (
          <PreprocessingRunSelection
            setSelectedPreprocessingRun={setSelectedPreprocessingRun}
            running={running}
            continueRun={continueRun}
            stopRun={stopRun}
          ></PreprocessingRunSelection>
        )}
        path="/preprocessing-runs/run-selection"
      ></Route>
    </Switch>
  );
}
