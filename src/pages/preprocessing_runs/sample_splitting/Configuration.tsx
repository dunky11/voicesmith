import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Button, Form, Input, Select } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  DatasetInterface,
  RunInterface,
  TextNormalizationInterface,
  TextNormalizationConfigInterface,
  SampleSplittingRunInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { notifySave } from "../../../utils";
import {
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_PREPROCESSING_NAMES_USED_CHANNEL,
  FETCH_DATASET_CANDIATES_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
} from "../../../channels";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
const { ipcRenderer } = window.require("electron");

const initialValues: {
  name: string;
  maximumWorkers: number;
  datasetID: number | null;
} = {
  name: "",
  maximumWorkers: -1,
  datasetID: null,
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
  run: SampleSplittingRunInterface | null;
}): ReactElement {
  const [names, setNames] = useState<string[]>([]);
  const isMounted = useRef(false);
  const [datasetsIsLoaded, setDatastsIsLoaded] = useState(false);
  const [configIsLoaded, setConfigIsLoaded] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInterface[]>([]);
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
      .invoke(UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL.IN, run.ID, values)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (run.stage === "not_started") {
            continueRun({
              ID: run.ID,
              type: "sampleSplittingRun",
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
    if (run === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL.IN, run.ID)
      .then((configuration: TextNormalizationInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (!configIsLoaded) {
          setConfigIsLoaded(true);
        }
        formRef.current?.setFieldsValue(configuration);
      });
  };

  const fetchNamesInUse = () => {
    if (run === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_PREPROCESSING_NAMES_USED_CHANNEL.IN, run.ID)
      .then((names: string[]) => {
        if (!isMounted.current) {
          return;
        }
        setNames(names);
      });
  };

  const fetchDatasets = () => {
    ipcRenderer
      .invoke(FETCH_DATASET_CANDIATES_CHANNEL.IN)
      .then((datasets: DatasetInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setDatasets(datasets);
        setDatastsIsLoaded(true);
      });
  };

  const getNextButtonText = () => {
    if (run === null || run.stage === "not_started") {
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
    if (run === null) {
      return;
    }
    fetchNamesInUse();
    fetchDatasets();
    fetchConfiguration();
  }, [run]);

  const disableNameEdit = !configIsLoaded || !datasetsIsLoaded;
  const disableElseEdit = disableNameEdit || run.stage !== "not_started";

  const disableNext = !configIsLoaded || !datasetsIsLoaded;
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
        <Form.Item
          label="Name"
          name="name"
          rules={[
            () => ({
              validator(_, value: string) {
                if (value.trim() === "") {
                  return Promise.reject(new Error("Please enter a name"));
                }
                if (names.includes(value)) {
                  return Promise.reject(
                    new Error("This name is already in use")
                  );
                }
                return Promise.resolve();
              },
            }),
          ]}
        >
          <Input disabled={disableNameEdit}></Input>
        </Form.Item>
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
        <Form.Item
          label="Dataset"
          name="datasetID"
          rules={[
            () => ({
              validator(_, value: string) {
                if (value === null) {
                  return Promise.reject(new Error("Please select a dataset"));
                }
                return Promise.resolve();
              },
            }),
          ]}
        >
          <Select disabled={disableElseEdit}>
            {datasets.map((dataset: DatasetInterface) => (
              <Select.Option
                value={dataset.ID}
                key={dataset.ID}
                disabled={dataset.referencedBy !== null}
              >
                {dataset.name}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>
      </Form>
    </RunCard>
  );
}
