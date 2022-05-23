import React, { useState, useRef, useEffect } from "react";
import { Button, Form, Input, Select, notification } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  DatasetInterface,
  RunInterface,
  CleaningRunConfigInterface,
  TextNormalizationInterface,
  TextNormalizationConfigInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { notifySave } from "../../../utils";
const { ipcRenderer } = window.require("electron");

const initialValues: TextNormalizationConfigInterface = {
  name: "",
  datasetID: null,
  language: "en",
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
    | "text_normalization"
    | "choose_samples"
    | "finished"
    | null;
}) {
  const [names, setNames] = useState<string[]>([]);
  const isMounted = useRef(false);
  const [datasetsIsLoaded, setDatastsIsLoaded] = useState(false);
  const [configIsLoaded, setConfigIsLoaded] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInterface[]>([]);
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
      .invoke("update-text-normalization-run-config", selectedID, values)
      .then((event: any) => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (stage === "not_started") {
            continueRun({
              ID: selectedID,
              type: "textNormalizationRun",
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
      .invoke("fetch-text-normalization-run-config", selectedID)
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
    if (selectedID === null) {
      return;
    }
    ipcRenderer
      .invoke("fetch-preprocessing-names-used", selectedID)
      .then((names: string[]) => {
        if (!isMounted.current) {
          return;
        }
        setNames(names);
      });
  };

  const fetchDatasets = () => {
    ipcRenderer
      .invoke("fetch-dataset-candidates")
      .then((datasets: DatasetInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setDatasets(datasets);
        setDatastsIsLoaded(true);
      });
  };

  const getNextButtonText = () => {
    if (selectedID === null || stage === "not_started") {
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
    fetchNamesInUse();
    fetchDatasets();
    fetchConfiguration();
  }, [selectedID]);

  const disableNameEdit = !configIsLoaded || !datasetsIsLoaded;
  const disableElseEdit = disableNameEdit || stage !== "not_started";

  const disableNext = !configIsLoaded || !datasetsIsLoaded;
  const disableDefaults = disableNext || stage != "not_started";

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
