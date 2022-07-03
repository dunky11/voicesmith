import React, { useState, useRef, useEffect, ReactElement } from "react";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch } from "react-redux";
import { cleaningRunInitialValues } from "../../../config";
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
import RunConfiguration from "../../../components/runs/RunConfiguration";
import DeviceInput from "../../../components/inputs/DeviceInput";
import { notifySave } from "../../../utils";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import SkipOnErrorInput from "../../../components/inputs/SkipOnErrorInput";
import MaximumWorkersInput from "../../../components/inputs/MaximumWorkersInput";
const { ipcRenderer } = window.require("electron");

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

  const onBack = () => {
    history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
  };

  const onNext = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const onDefaults = () => {
    formRef.current?.setFieldsValue({
      ...cleaningRunInitialValues,
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
      ...cleaningRunInitialValues,
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
    <RunConfiguration
      title="Configuration"
      forms={
        <>
          <MaximumWorkersInput disabled={initialIsLoading} />
          <DeviceInput disabled={initialIsLoading} />
          <SkipOnErrorInput disabled={initialIsLoading} />
          <DatasetInput disabled={hasStarted || initialIsLoading} />
        </>
      }
      hasStarted={run.stage !== "not_started"}
      isDisabled={initialIsLoading}
      onBack={onBack}
      onDefaults={onDefaults}
      onSave={onSave}
      onNext={onNext}
      formRef={formRef}
      initialValues={cleaningRunInitialValues}
      onFinish={onFinish}
      fetchNames={() => fetchNames(run.ID)}
    />
  );
}
