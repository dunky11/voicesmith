import React, { useState, useRef, useEffect } from "react";
import { Button, Form, Input, Select, notification } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  DatasetInterface,
  RunInterface,
  CleaningRunConfigInterface,
} from "../../../interfaces";
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
      .invoke("update-cleaning-run-config", selectedID, values)
      .then((event: any) => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (stage === "not_started") {
            continueRun({
              ID: selectedID,
              type: "dSCleaning",
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
      .invoke("fetch-cleaning-run-config", selectedID)
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
        style={{ padding: "16px 16px 0px 16px" }}
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
      </Form>
    </RunCard>
  );
}
