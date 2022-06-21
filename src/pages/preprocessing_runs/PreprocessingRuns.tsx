import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, Route, useHistory } from "react-router-dom";
import { PREPROCESSING_RUNS_ROUTE } from "../../routes";
import { RunInterface } from "../../interfaces";
import PreprocessingRunSelection from "./PreprocessingRunSelection";
import TextNormalization from "./text_normalization/TextNormalization";
import DatasetCleaning from "./dataset_cleaning/DatasetCleaning";
import SampleSplitting from "./sample_splitting/SampleSplitting";
import { FETCH_PREPROCESSING_NAMES_USED_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");

export const fetchNames = (runID: number): Promise<string[]> => {
  return new Promise((resolve) => {
    ipcRenderer
      .invoke(FETCH_PREPROCESSING_NAMES_USED_CHANNEL.IN, runID)
      .then((names: string[]) => {
        resolve(names);
      });
  });
};

export default function PreprcocessingRuns(): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [selectedPreprocessingRun, setSelectedPreprocessingRun] =
    useState<RunInterface | null>(null);

  useEffect(() => {
    if (selectedPreprocessingRun === null) {
      return;
    }
    switch (selectedPreprocessingRun.type) {
      case "textNormalizationRun":
        history.push(
          PREPROCESSING_RUNS_ROUTE.TEXT_NORMALIZATION.CONFIGURATION.ROUTE
        );
        break;
      case "cleaningRun":
        history.push(
          PREPROCESSING_RUNS_ROUTE.DATASET_CLEANING.CONFIGURATION.ROUTE
        );
        break;
      case "sampleSplittingRun":
        history.push(
          PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.CONFIGURATION.ROUTE
        );
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
        render={() =>
          selectedPreprocessingRun === null ? (
            <></>
          ) : (
            <TextNormalization
              preprocessingRun={selectedPreprocessingRun}
            ></TextNormalization>
          )
        }
        path={PREPROCESSING_RUNS_ROUTE.TEXT_NORMALIZATION.ROUTE}
      ></Route>
      <Route
        path={PREPROCESSING_RUNS_ROUTE.DATASET_CLEANING.ROUTE}
        render={() =>
          selectedPreprocessingRun === null ? (
            <></>
          ) : (
            <DatasetCleaning
              preprocessingRun={selectedPreprocessingRun}
            ></DatasetCleaning>
          )
        }
      ></Route>
      <Route
        render={() =>
          selectedPreprocessingRun === null ? (
            <></>
          ) : (
            <SampleSplitting
              preprocessingRun={selectedPreprocessingRun}
            ></SampleSplitting>
          )
        }
        path={PREPROCESSING_RUNS_ROUTE.SAMPLE_SPLITTING.ROUTE}
      ></Route>
      <Route
        render={() => (
          <PreprocessingRunSelection
            setSelectedPreprocessingRun={setSelectedPreprocessingRun}
          ></PreprocessingRunSelection>
        )}
        path={PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE}
      ></Route>
    </Switch>
  );
}
