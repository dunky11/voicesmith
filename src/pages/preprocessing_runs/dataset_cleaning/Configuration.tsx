import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form, Input, Select, notification } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  UPDATE_CLEANING_RUN_CONFIG_CHANNEL,
  FETCH_CLEANING_RUN_CONFIG_CHANNEL,
} from "../../../channels";
import { fetchNames } from "../PreprocessingRuns";
import { RunInterface, CleaningRunConfigInterface } from "../../../interfaces";
import DatasetInput from "../../../components/inputs/DatasetInput";
import NameInput from "../../../components/inputs/NameInput";
import RunCard from "../../../components/cards/RunCard";
import { notifySave } from "../../../utils";
const { ipcRenderer } = window.require("electron");

const initialValues: CleaningRunConfigInterface = {
  name: "",
  datasetID: null,
};

export default function Configuration({
  onStepChange,
  selectedID,
  running,
  continueRun,
  stage,
}: {
  onStepChange: (current: number) => void;
  selectedID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stage:
    | "not_started"
    | "preparing"
    | "gen_file_embeddings"
    | "detect_outliers"
    | "choose_samples"
    | "apply_changes"
    | "finished"
    | null;
}): ReactElement {
  const isMounted = useRef(false);
  const [configIsLoaded, setConfigIsLoaded] = useState(false);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef = useRef<FormInstance | null>();

  const onBackClick = () => {
    history.push("/preprocessing-runs/run-selection");
  };

  const onNextClick = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const onDefaults = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = false;
    formRef.current.submit();
  };

  const onSave = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = false;
    formRef.current.submit();
  };

  const onFinish = () => {
    const values: CleaningRunConfigInterface = {
      ...initialValues,
      ...formRef.current?.getFieldsValue(),
    };

    ipcRenderer
      .invoke(UPDATE_CLEANING_RUN_CONFIG_CHANNEL.IN, selectedID, values)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (stage === "not_started") {
            continueRun({
              ID: selectedID,
              type: "dSCleaningRun",
            });
          }
          onStepChange(1);
          navigateNextRef.current = false;
        } else {
          notifySave();
        }
      });
  };

  const fetchConfiguration = () => {
    if (selectedID === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_CLEANING_RUN_CONFIG_CHANNEL.IN, selectedID)
      .then((configuration: CleaningRunConfigInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (!configIsLoaded) {
          setConfigIsLoaded(true);
        }
        formRef.current?.setFieldsValue(configuration);
      });
  };

  const getNextButtonText = () => {
    if (selectedID === null || stage == "not_started") {
      return "Save and Start Run";
    }
    return "Save and Next";
  };

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (selectedID === null) {
      return;
    }
    fetchConfiguration();
  }, [selectedID]);

  const disableNameEdit = !configIsLoaded;
  const disableElseEdit = disableNameEdit || stage !== "not_started";

  const disableNext = !configIsLoaded;
  const disableDefaults = disableNext || stage != "not_started";

  return (
    <RunCard
      title="Configure the Cleaning Run"
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button disabled={disableDefaults} onClick={onDefaults}>
          Reset to Default
        </Button>,
        <Button onClick={onSave}>Save</Button>,
        <Button type="primary" disabled={disableNext} onClick={onNextClick}>
          {getNextButtonText()}
        </Button>,
      ]}
    >
      <Form
        layout="vertical"
        ref={(node) => {
          formRef.current = node;
        }}
        initialValues={initialValues}
        onFinish={onFinish}
      >
        <NameInput
          fetchNames={() => {
            return fetchNames(selectedID);
          }}
          disabled={disableNameEdit}
        ></NameInput>
        <DatasetInput disabled={disableElseEdit} />
      </Form>
    </RunCard>
  );
}
