import React, { useState, useRef, useEffect, ReactElement } from "react";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch } from "react-redux";
import RunConfiguration from "../../../components/runs/RunConfiguration";
import { textNormalizationRunInitialValues } from "../../../config";
import {
  TextNormalizationRunInterface,
  TextNormalizationRunConfigInterface,
} from "../../../interfaces";
import DatasetInput from "../../../components/inputs/DatasetInput";
import { fetchNames } from "../PreprocessingRuns";
import { notifySave } from "../../../utils";
import {
  UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
  FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
} from "../../../channels";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import { addToQueue } from "../../../features/runManagerSlice";
const { ipcRenderer } = window.require("electron");

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
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
      ...textNormalizationRunInitialValues,
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
    const values: TextNormalizationRunConfigInterface = {
      ...textNormalizationRunInitialValues,
      ...formRef.current?.getFieldsValue(),
    };

    ipcRenderer
      .invoke(UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL.IN, run.ID, values)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (run.stage === "not_started") {
            dispatch(
              addToQueue({
                ID: run.ID,
                type: "textNormalizationRun",
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
      .invoke(FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL.IN, run.ID)
      .then((configuration: TextNormalizationRunInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (initialIsLoading) {
          setInitialIsLoading(false);
        }
        formRef.current?.setFieldsValue(configuration);
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
      title="Configure the Text Normalization Run"
      forms={<DatasetInput disabled={initialIsLoading || hasStarted} />}
      hasStarted={run.stage !== "not_started"}
      isDisabled={initialIsLoading}
      onBack={onBack}
      onDefaults={onDefaults}
      onSave={onSave}
      onNext={onNext}
      formRef={formRef}
      initialValues={textNormalizationRunInitialValues}
      onFinish={onFinish}
      fetchNames={() => fetchNames(run.ID)}
    />
  );
}
