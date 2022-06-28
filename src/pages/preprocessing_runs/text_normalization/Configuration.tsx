import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { textNormalizationRunInitialValues } from "../../../config";
import {
  TextNormalizationRunInterface,
  TextNormalizationRunConfigInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import DatasetInput from "../../../components/inputs/DatasetInput";
import NameInput from "../../../components/inputs/NameInput";
import { fetchNames } from "../PreprocessingRuns";
import { notifySave } from "../../../utils";
import {
  UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
  FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL,
} from "../../../channels";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import { useDispatch } from "react-redux";
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
        if (!initialIsLoading) {
          setInitialIsLoading(false);
        }
        formRef.current?.setFieldsValue(configuration);
      });
  };

  const getNextButtonText = () => {
    if (run.stage === "not_started") {
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
      title="Configure the Text Normalization Run"
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
        initialValues={textNormalizationRunInitialValues}
        onFinish={onFinish}
      >
        <NameInput
          disabled={initialIsLoading}
          fetchNames={() => {
            return fetchNames(run.ID);
          }}
        />
        <DatasetInput disabled={initialIsLoading || hasStarted} />
      </Form>
    </RunCard>
  );
}
