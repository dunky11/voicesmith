import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form, Select } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch } from "react-redux";
import { addToQueue } from "../../../features/runManagerSlice";
import {
  UPDATE_CLEANING_RUN_CONFIG_CHANNEL,
  FETCH_CLEANING_RUNS_CHANNEL,
} from "../../../channels";
import { fetchNames } from "../PreprocessingRuns";
import {
  CleaningRunConfigInterface,
  CleaningRunInterface,
} from "../../../interfaces";
import DatasetInput from "../../../components/inputs/DatasetInput";
import NameInput from "../../../components/inputs/NameInput";
import DeviceInput from "../../../components/inputs/DeviceInput";
import RunCard from "../../../components/cards/RunCard";
import { notifySave } from "../../../utils";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import SkipOnErrorInput from "../../../components/inputs/SkipOnErrorInput";
import MaximumWorkersInput from "../../../components/inputs/MaximumWorkersInput";
const { ipcRenderer } = window.require("electron");

const initialValues: CleaningRunConfigInterface = {
  name: "",
  datasetID: null,
  datasetName: null,
  skipOnError: true,
  device: "CPU",
  maximumWorkers: -1,
};

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: CleaningRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const [initialIsLoading, setInitialIsLoading] = useState(true);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef = useRef<FormInstance | null>();

  const onBackClick = () => {
    history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
  };

  const onNextClick = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const onDefaults = () => {
    formRef.current?.setFieldsValue({
      ...initialValues,
      datasetID: formRef.current.getFieldValue("datasetID"),
      datasetName: formRef.current.getFieldValue("datasetName"),
      name: formRef.current.getFieldValue("name"),
    });
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
      .invoke(UPDATE_CLEANING_RUN_CONFIG_CHANNEL.IN, run.ID, values)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (run.stage === "not_started") {
            dispatch(
              addToQueue({
                ID: run.ID,
                type: "cleaningRun",
                name: run.name,
              })
            );
          }
          onStepChange(1);
          navigateNextRef.current = false;
        } else {
          notifySave();
        }
      });
  };

  const fetchConfiguration = () => {
    ipcRenderer
      .invoke(FETCH_CLEANING_RUNS_CHANNEL.IN, run.ID)
      .then((runs: CleaningRunInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        if (initialIsLoading) {
          setInitialIsLoading(false);
        }
        formRef.current?.setFieldsValue(runs[0].configuration);
      });
  };

  const getNextButtonText = () => {
    if (run.stage == "not_started") {
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
    fetchConfiguration();
  }, []);

  const hasStarted = run.stage !== "not_started";

  return (
    <RunCard
      title="Configure the Cleaning Run"
      buttons={[
        <Button onClick={onBackClick}>Back</Button>,
        <Button disabled={initialIsLoading} onClick={onDefaults}>
          Reset to Default
        </Button>,
        <Button disabled={initialIsLoading} onClick={onSave}>
          Save
        </Button>,
        <Button
          type="primary"
          disabled={initialIsLoading}
          onClick={onNextClick}
        >
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
            return fetchNames(run.ID);
          }}
          disabled={initialIsLoading}
        ></NameInput>
        <MaximumWorkersInput disabled={initialIsLoading} />
        <DeviceInput disabled={initialIsLoading} />
        <SkipOnErrorInput disabled={initialIsLoading} />
        <DatasetInput disabled={hasStarted || initialIsLoading} />
      </Form>
    </RunCard>
  );
}
