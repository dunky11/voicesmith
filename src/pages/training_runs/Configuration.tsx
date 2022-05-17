import React, { useState, useRef, useEffect } from "react";
import {
  Card,
  Button,
  Form,
  Input,
  Collapse,
  InputNumber,
  Checkbox,
  Select,
  Alert,
  Typography,
  notification,
} from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import {
  ConfigurationInterface,
  DatasetInterface,
  RunInterface,
} from "../../interfaces";
import { SERVER_URL, trainingRunInitialValues } from "../../config";
const { ipcRenderer, shell } = window.require("electron");

const notifySave = () => {
  notification["success"]({
    message: "Your configuration has been saved",
    placement: "top",
  });
};

export default function Configuration({
  onStepChange,
  setSelectedTrainingRunID,
  selectedTrainingRunID,
  running,
  continueRun,
  stage,
}: {
  onStepChange: (current: number) => void;
  setSelectedTrainingRunID: (selectedTrainingRunID: number) => void;
  selectedTrainingRunID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stage:
    | "not_started"
    | "preprocessing"
    | "acoustic_fine_tuning"
    | "ground_truth_alignment"
    | "vocoder_fine_tuning"
    | "save_model"
    | "finished"
    | null;
}) {
  const [modelNames, setModelNames] = useState<string[]>([]);
  const isMounted = useRef(false);
  const [datasetsIsLoaded, setDatastsIsLoaded] = useState(false);

  const [configIsLoaded, setConfigIsLoaded] = useState(false);
  const [cudaStateIsLoaded, setCudaStateIsLoaded] = useState(false);
  const [cudaIsAvailable, setCudaIsAvailable] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInterface[]>([]);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef = useRef<FormInstance | null>();

  const onBackClick = () => {
    history.push("/training-runs/run-selection");
  };

  const afterUpdate = () => {
    if (!isMounted.current) {
      return;
    }
    if (navigateNextRef.current) {
      continueRun({
        ID: selectedTrainingRunID,
        type: "trainingRun",
      });
      onStepChange(1);
    } else {
      notifySave();
    }
  };

  const onFinish = () => {
    const values: ConfigurationInterface = {
      ...trainingRunInitialValues,
      ...formRef.current?.getFieldsValue(),
    };

    if (selectedTrainingRunID === null) {
      ipcRenderer.invoke("create-training-run", values).then(afterUpdate);
    } else {
      ipcRenderer
        .invoke("update-training-run-config", values, selectedTrainingRunID)
        .then(afterUpdate);
    }
  };

  const onSave = () => {
    navigateNextRef.current = false;
    formRef.current.submit();
  };

  const onNextClick = () => {
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const onDefaults = () => {
    const values = {
      ...trainingRunInitialValues,
      name: formRef.current.getFieldValue("name"),
      device: cudaIsAvailable ? "GPU" : "CPU",
    };
    formRef.current?.setFieldsValue(values);
  };

  const fetchConfiguration = () => {
    ipcRenderer
      .invoke("fetch-training-run-configuration", selectedTrainingRunID)
      .then((configuration: ConfigurationInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (!configIsLoaded) {
          setConfigIsLoaded(true);
        }
        const config = {
          ...configuration,
          device: cudaIsAvailable ? configuration.device : "CPU",
        };
        formRef.current?.setFieldsValue(config);
      });
  };

  const fetchNamesInUse = () => {
    ipcRenderer
      .invoke("fetch-training-run-names", selectedTrainingRunID)
      .then((names: string[]) => {
        if (!isMounted.current) {
          return;
        }
        setModelNames(names);
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

  const fetchIsCudaAvailable = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", `${SERVER_URL}/is-cuda-available`);
    ajax.onload = () => {
      if (!isMounted.current) {
        return;
      }
      const response = JSON.parse(ajax.responseText);
      setCudaIsAvailable(response.available);
      setCudaStateIsLoaded(true);
    };
    ajax.send();
  };

  useEffect(() => {
    isMounted.current = true;
    fetchNamesInUse();
    fetchDatasets();
    fetchIsCudaAvailable();
    fetchConfiguration();
    return () => {
      isMounted.current = false;
    };
  }, []);

  const disableEdit =
    !cudaStateIsLoaded || !configIsLoaded || !datasetsIsLoaded;

  const disableNext = disableEdit;
  const disableDefaults =
    !configIsLoaded || (stage != "not_started" && stage != null);

  return (
    <Card
      title="Configure the Training Run"
      style={{ height: "100%" }}
      bodyStyle={{ width: "100%", padding: "0 0 32px 0" }}
      className="dataset-card-wrapper"
      actions={[
        <div
          key="next-button-wrapper"
          style={{
            display: "flex",
            justifyContent: "flex-end",
            marginRight: 24,
          }}
        >
          <Button style={{ marginRight: 8 }} onClick={onBackClick}>
            Back
          </Button>
          <Button
            style={{ marginRight: 8 }}
            disabled={disableDefaults}
            onClick={onDefaults}
          >
            Reset to Default
          </Button>
          <Button style={{ marginRight: 8 }} onClick={onSave}>
            Save
          </Button>
          <Button type="primary" disabled={disableNext} onClick={onNextClick}>
            Save and Start Training
          </Button>
        </div>,
      ]}
    >
      <Form
        layout="vertical"
        ref={(node) => {
          formRef.current = node;
        }}
        onFinish={onFinish}
        style={{ padding: "16px 16px 0px 16px" }}
        initialValues={trainingRunInitialValues}
      >
        <Form.Item
          label="Model Name"
          name="name"
          rules={[
            () => ({
              validator(_, value: string) {
                if (value.trim() === "") {
                  return Promise.reject(new Error("Please enter a name"));
                }
                if (modelNames.includes(value)) {
                  return Promise.reject(
                    new Error("This name is already in use")
                  );
                }
                return Promise.resolve();
              },
            }),
          ]}
        >
          <Input disabled={disableEdit}></Input>
        </Form.Item>
        <Form.Item
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
          label="Dataset"
          name="datasetID"
        >
          <Select disabled={disableEdit}>
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
        <Form.Item label="Train On" name="device">
          <Select disabled={disableEdit}>
            <Select.Option value="CPU">CPU</Select.Option>
            <Select.Option value="GPU" disabled={!cudaIsAvailable}>
              GPU
            </Select.Option>
          </Select>
        </Form.Item>
        {!cudaIsAvailable && (
          <Alert
            style={{ marginBottom: 24 }}
            message={
              <Typography.Text>
                No CUDA supported GPU was detected. While you can train on CPU,
                training on GPU is highly recommended since training on CPU will
                most likely take days. If you want to train on GPU{" "}
                <a
                  onClick={() => {
                    shell.openExternal(
                      "https://developer.nvidia.com/cuda-gpus"
                    );
                  }}
                >
                  please make sure it has CUDA support
                </a>{" "}
                and it&apos;s driver is up to date. Afterwards restart the app.
              </Typography.Text>
            }
            type="warning"
          />
        )}
        <Collapse style={{ width: "100%" }}>
          <Collapse.Panel header="Preprocessing" key="preprocessing">
            <Form.Item label="Validation Size" name="validationSize">
              <InputNumber
                disabled={disableEdit}
                step={0.01}
                min={0}
                max={100.0}
                addonAfter="%"
              ></InputNumber>
            </Form.Item>
            <Form.Item
              rules={[
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (value > getFieldValue("maxSeconds")) {
                      return Promise.reject(
                        new Error(
                          "Minimum seconds must be smaller than maximum seconds"
                        )
                      );
                    }
                    return Promise.resolve();
                  },
                }),
              ]}
              label="Minimum Seconds"
              name="minSeconds"
              dependencies={["maxSeconds"]}
            >
              <InputNumber
                disabled={disableEdit}
                step={0.1}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item
              label="Maximum Seconds"
              rules={[
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (value <= getFieldValue("minSeconds")) {
                      return Promise.reject(
                        new Error(
                          "Maximum seconds must be greater than minimum seconds"
                        )
                      );
                    }
                    return Promise.resolve();
                  },
                }),
              ]}
              dependencies={["minSeconds"]}
              name="maxSeconds"
            >
              <InputNumber
                disabled={disableEdit}
                step={0.1}
                min={0}
                max={15}
              ></InputNumber>
            </Form.Item>
            <Form.Item name="useAudioNormalization" valuePropName="checked">
              <Checkbox disabled={disableEdit}>
                Apply Audio Normalization
              </Checkbox>
            </Form.Item>
          </Collapse.Panel>
          <Collapse.Panel header="Acoustic Model" key="acoustic model">
            <Form.Item
              rules={[
                () => ({
                  validator(_, value) {
                    if (value === 0) {
                      return Promise.reject(
                        new Error("Learning rate must be greater than zero")
                      );
                    }
                    return Promise.resolve();
                  },
                }),
              ]}
              label="Learning Rate"
              name="acousticLearningRate"
            >
              <InputNumber
                disabled={disableEdit}
                step={0.001}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Training Steps" name="acousticTrainingIterations">
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Batch Size" name="acousticBatchSize">
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={1}
              ></InputNumber>
            </Form.Item>
            <Form.Item
              label="Gradient Accumulation Steps"
              name="acousticGradAccumSteps"
            >
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={1}
              ></InputNumber>
            </Form.Item>
            <Form.Item
              label="Run Validation Every"
              name="acousticValidateEvery"
            >
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={10}
                min={0}
                addonAfter="steps"
              ></InputNumber>
            </Form.Item>
            <Form.Item
              rules={[
                ({ getFieldValue }) => ({
                  validator(_, value) {
                    if (value > getFieldValue("acousticTrainingIterations")) {
                      return Promise.reject(
                        new Error("Cannot be smaller than training steps")
                      );
                    }
                    return Promise.resolve();
                  },
                }),
              ]}
              label="Train Only Speaker Embeds Until"
              name="onlyTrainSpeakerEmbUntil"
            >
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={10}
                min={0}
                addonAfter="steps"
              ></InputNumber>
            </Form.Item>
          </Collapse.Panel>

          <Collapse.Panel header="Vocoder" key="vocoder">
            <Form.Item
              rules={[
                () => ({
                  validator(_, value) {
                    if (value === 0) {
                      return Promise.reject(
                        new Error("Learning rate must be greater than zero")
                      );
                    }
                    return Promise.resolve();
                  },
                }),
              ]}
              label="Learning Rate"
              name="vocoderLearningRate"
            >
              <InputNumber
                disabled={disableEdit}
                step={0.001}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item
              label="Training Iterations"
              name="vocoderTrainingIterations"
            >
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Batch Size" name="vocoderBatchSize">
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={1}
              ></InputNumber>
            </Form.Item>
            <Form.Item
              label="Gradient Accumulation Steps"
              name="vocoderGradAccumSteps"
            >
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={1}
                min={1}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Run Validation Every" name="vocoderValidateEvery">
              <InputNumber
                precision={0}
                disabled={disableEdit}
                step={10}
                min={0}
                addonAfter="steps"
              ></InputNumber>
            </Form.Item>
          </Collapse.Panel>
        </Collapse>
      </Form>
    </Card>
  );
}
