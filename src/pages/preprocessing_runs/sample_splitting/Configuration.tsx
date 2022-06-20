import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form, Select } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { RunInterface, SampleSplittingRunInterface } from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { notifySave } from "../../../utils";
import {
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
} from "../../../channels";
import DeviceInput from "../../../components/inputs/DeviceInput";
import DatasetInput from "../../../components/inputs/DatasetInput";
import NameInput from "../../../components/inputs/NameInput";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import { fetchNames } from "../PreprocessingRuns";
const { ipcRenderer } = window.require("electron");

const initialValues: {
  name: string;
  maximumWorkers: number;
  datasetID: number | null;
  device: "CPU" | "GPU";
} = {
  name: "",
  maximumWorkers: -1,
  datasetID: null,
  device: "CPU",
};

export default function Configuration({
  onStepChange,
  running,
  continueRun,
  run,
}: {
  onStepChange: (current: number) => void;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  run: SampleSplittingRunInterface;
}): ReactElement {
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
    const values: SampleSplittingRunInterface = {
      ...initialValues,
      ...formRef.current?.getFieldsValue(),
      ID: run.ID,
    };

    ipcRenderer
      .invoke(UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL.IN, values)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (run.stage === "not_started") {
            continueRun({
              ID: run.ID,
              type: "sampleSplittingRun",
              name: run.name,
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
    ipcRenderer
      .invoke(FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL.IN, run.ID)
      .then((runs: SampleSplittingRunInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        if (!configIsLoaded) {
          setConfigIsLoaded(true);
        }
        formRef.current?.setFieldsValue(runs[0]);
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
  const disableDefaults = disableNext || run.stage != "not_started";

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
        <Form.Item label="Maximum Number of Workers" name="maximumWorkers">
          <Select style={{ width: 200 }}>
            <Select.Option value={-1}>Auto</Select.Option>
            {Array.from(Array(64 + 1).keys())
              .slice(1)
              .map((el) => (
                <Select.Option key={el} value={el}>
                  {el}
                </Select.Option>
              ))}
          </Select>
        </Form.Item>
        <DatasetInput disabled={disableElseEdit} />
        <DeviceInput disabled={disableElseEdit} />
      </Form>
    </RunCard>
  );
}
