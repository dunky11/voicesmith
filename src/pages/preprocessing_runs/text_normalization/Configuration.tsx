import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form, Input, Select } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  RunInterface,
  TextNormalizationRunInterface,
  TextNormalizationConfigInterface,
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
import { setIsRunning, addToQueue } from "../../../features/runManagerSlice";
const { ipcRenderer } = window.require("electron");

const initialValues: TextNormalizationConfigInterface = {
  name: "",
  datasetID: null,
  language: "en",
};

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const [configIsLoaded, setConfigIsLoaded] = useState(false);
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
    const values: TextNormalizationConfigInterface = {
      ...initialValues,
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
            dispatch(setIsRunning(true));
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
        if (!configIsLoaded) {
          setConfigIsLoaded(true);
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

  const disableNameEdit = !configIsLoaded;
  const disableElseEdit = disableNameEdit || run.stage !== "not_started";

  const disableNext = !configIsLoaded;
  const disableDefaults = disableNext || run.stage !== "not_started";

  return (
    <RunCard
      title="Configure the Text Normalization Run"
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
          disabled={disableNameEdit}
          fetchNames={() => {
            return fetchNames(run.ID);
          }}
        />
        <DatasetInput disabled={disableElseEdit} />
        <Form.Item
          label="Language"
          name="language"
          rules={[
            () => ({
              validator(_, value: string) {
                if (value === null) {
                  return Promise.reject(new Error("Please select a language"));
                }
                return Promise.resolve();
              },
            }),
          ]}
        >
          <Select disabled={disableElseEdit}>
            <Select.Option value="en" key="en">
              English
            </Select.Option>
            <Select.Option value="es" key="es">
              Spanish
            </Select.Option>
            <Select.Option value="de" key="de">
              German
            </Select.Option>
            <Select.Option value="ru" key="ru">
              Russian
            </Select.Option>
          </Select>
        </Form.Item>
      </Form>
    </RunCard>
  );
}
