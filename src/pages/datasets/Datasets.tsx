import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Switch, Route, useHistory } from "react-router-dom";
import Dataset from "./Dataset";
import DatasetSelection from "./DatasetSelection";
import { DATASETS_ROUTE } from "../../routes";

export default function Datasets({
  setNavIsDisabled,
}: {
  setNavIsDisabled: (isDisabled: boolean) => void;
}): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [selectedDatasetID, setSelectedDatasetID] = useState<number | null>(
    null
  );

  const passSelectedSpeakerID = (ID: number | null) => {
    if (ID === selectedDatasetID && ID !== null) {
      history.push(DATASETS_ROUTE.EDIT.ROUTE);
    } else {
      setSelectedDatasetID(ID);
    }
  };

  useEffect(() => {
    if (selectedDatasetID === null) {
      return;
    }
    history.push(DATASETS_ROUTE.EDIT.ROUTE);
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
            setNavIsDisabled={setNavIsDisabled}
          ></DatasetSelection>
        )}
        path={DATASETS_ROUTE.SELECTION.ROUTE}
      ></Route>
      <Route
        render={() => <Dataset datasetID={selectedDatasetID}></Dataset>}
        path={DATASETS_ROUTE.EDIT.ROUTE}
      ></Route>
    </Switch>
  );
}
