import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, Route, useHistory } from "react-router-dom";
import Dataset from "./Dataset";
import DatasetSelection from "./DatasetSelection";

export default function Datasets(): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [selectedDatasetID, setSelectedDatasetID] = useState<number | null>(
    null
  );

  const passSelectedSpeakerID = (ID: number | null) => {
    if (ID === selectedDatasetID && ID !== null) {
      history.push("/datasets/dataset-edit");
    } else {
      setSelectedDatasetID(ID);
    }
  };

  useEffect(() => {
    if (selectedDatasetID === null) {
      return;
    }
    history.push("/datasets/dataset-edit");
  }, [selectedDatasetID]);

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
          <DatasetSelection
            setSelectedDatasetID={passSelectedSpeakerID}
          ></DatasetSelection>
        )}
        path="/datasets/dataset-selection"
      ></Route>
      <Route
        render={() => <Dataset datasetID={selectedDatasetID}></Dataset>}
        path="/datasets/dataset-edit"
      ></Route>
    </Switch>
  );
}
