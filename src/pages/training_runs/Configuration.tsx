import React, { useState, useRef, useEffect, ReactElement } from "react";
import {
  Button,
  Form,
  Input,
  Collapse,
  InputNumber,
  Checkbox,
  Select,
} from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import {
  RunInterface,
  TrainingRunConfigInterface,
  TrainingRunInterface,
} from "../../interfaces";
import { trainingRunInitialValues } from "../../config";
import { notifySave } from "../../utils";
import RunCard from "../../components/cards/RunCard";
import SkipOnErrorInput from "../../components/inputs/SkipOnErrorInput";
import DeviceInput from "../../components/inputs/DeviceInput";
import DatasetInput from "../../components/inputs/DatasetInput";
import NameInput from "../../components/inputs/NameInput";
import MaximumWorkersInput from "../../components/inputs/MaximumWorkersInput";
import {
  UPDATE_TRAINING_RUN_CHANNEL,
  FETCH_TRAINING_RUN_NAMES_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL_TYPES,
} from "../../channels";
import { TRAINING_RUNS_ROUTE } from "../../routes";
import { addToQueue } from "../../features/runManagerSlice";
const { ipcRenderer } = window.require("electron");

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TrainingRunInterface;
}): ReactElement {
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const [initialIsLoading, setInitialIsLoading] = useState(true);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef = useRef<FormInstance | null>();

  const onBackClick = () => {
    history.push(TRAINING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
  };

  const afterUpdate = () => {
    if (!isMounted.current) {
      return;
    }
    if (navigateNextRef.current) {
      dispatch(
        addToQueue({
          ID: run.ID,
          type: "trainingRun",
          name: run.name,
        })
      );
      onStepChange(1);
    } else {
      notifySave();
    }
  };

  const onFinish = () => {
    const values: TrainingRunConfigInterface = {
      ...trainingRunInitialValues,
      ...formRef.current?.getFieldsValue(),
    };
    ipcRenderer
      .invoke(UPDATE_TRAINING_RUN_CHANNEL.IN, {
        ...{ ...run, configuration: values },
      })
      .then(afterUpdate);
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
      datasetName: formRef.current.getFieldValue("datasetName"),
      datasetID: formRef.current.getFieldValue("datasetID"),
      name: formRef.current.getFieldValue("name"),
    };
    formRef.current?.setFieldsValue(values);
  };

  const fetchConfiguration = () => {
    const args: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      withStatistics: false,
      ID: run.ID,
    };
    ipcRenderer
      .invoke(FETCH_TRAINING_RUNS_CHANNEL.IN, args)
      .then((runs: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        if (initialIsLoading) {
          setInitialIsLoading(false);
        }
        formRef.current?.setFieldsValue(runs[0].configuration);
      });
  };

  const fetchNames = async (): Promise<string[]> => {
    return new Promise((resolve) => {
      ipcRenderer
        .invoke(FETCH_TRAINING_RUN_NAMES_CHANNEL.IN, run.ID)
        .then((names: string[]) => {
          resolve(names);
        });
    });
  };

  useEffect(() => {
    isMounted.current = true;
    fetchConfiguration();

    return () => {
      isMounted.current = false;
    };
  }, []);

  const hasStarted = run.stage !== "not_started";

  return (
    <RunCard
      title="Configure the Training Run"
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
          {running !== null &&
          running.type === "trainingRun" &&
          running.ID === run.ID
            ? "Save and Next"
            : "Save and Start Training"}
        </Button>,
      ]}
    >
      <Form
        layout="vertical"
        ref={(node) => {
          formRef.current = node;
        }}
        onFinish={onFinish}
        initialValues={trainingRunInitialValues}
      >
        <NameInput fetchNames={fetchNames} disabled={initialIsLoading} />
        <SkipOnErrorInput disabled={initialIsLoading} />
        <DatasetInput disabled={initialIsLoading || hasStarted} />
        <DeviceInput disabled={initialIsLoading} />
        <Collapse style={{ width: "100%" }}>
          <Collapse.Panel header="Preprocessing" key="preprocessing">
            <Form.Item label="Validation Size" name="validationSize">
              <InputNumber
                disabled={initialIsLoading}
                step={0.01}
                min={0}
                max={100.0}
                addonAfter="%"
              ></InputNumber>
            </Form.Item>
            <MaximumWorkersInput disabled={initialIsLoading} />
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
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
                step={0.1}
                min={0}
                max={15}
              ></InputNumber>
            </Form.Item>
            <Form.Item name="useAudioNormalization" valuePropName="checked">
              <Checkbox disabled={initialIsLoading}>
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
                disabled={initialIsLoading}
                step={0.001}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Training Steps" name="acousticTrainingIterations">
              <InputNumber
                precision={0}
                disabled={initialIsLoading}
                step={1}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Batch Size" name="acousticBatchSize">
              <InputNumber
                precision={0}
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
                step={1}
                min={0}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Batch Size" name="vocoderBatchSize">
              <InputNumber
                precision={0}
                disabled={initialIsLoading}
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
                disabled={initialIsLoading}
                step={1}
                min={1}
              ></InputNumber>
            </Form.Item>
            <Form.Item label="Run Validation Every" name="vocoderValidateEvery">
              <InputNumber
                precision={0}
                disabled={initialIsLoading}
                step={10}
                min={0}
                addonAfter="steps"
              ></InputNumber>
            </Form.Item>
          </Collapse.Panel>
        </Collapse>
      </Form>
    </RunCard>
  );
}
