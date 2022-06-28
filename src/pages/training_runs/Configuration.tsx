import React, { useState, useRef, useEffect, ReactElement } from "react";
import {
  Button,
  Form,
  Collapse,
  InputNumber,
  Checkbox,
  Typography,
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
import TrainingStepsInput from "../../components/inputs/TrainingStepsInput";
import LearningRateInput from "../../components/inputs/LearningRateInput";
import SkipOnErrorInput from "../../components/inputs/SkipOnErrorInput";
import DeviceInput from "../../components/inputs/DeviceInput";
import DatasetInput from "../../components/inputs/DatasetInput";
import NameInput from "../../components/inputs/NameInput";
import MaximumWorkersInput from "../../components/inputs/MaximumWorkersInput";
import AlignmentBatchSizeInput from "../../components/inputs/AlignmentBatchSizeInput";
import HelpIcon from "../../components/help/HelpIcon";
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
            <Form.Item
              label={
                <Typography.Text>
                  Validation Size
                  <HelpIcon
                    content={
                      <Typography>
                        Percentage of samples in the dataset that will be
                        randomly chosen and put into the validation dataset. The
                        validation dataset is used for calculating the
                        validation losses. Calculating the validation losses is
                        important to measure the level of overfitting.
                      </Typography>
                    }
                    style={{ marginLeft: 8 }}
                  />
                </Typography.Text>
              }
              name="validationSize"
            >
              <InputNumber
                disabled={initialIsLoading}
                step={0.01}
                min={0}
                max={100.0}
                addonAfter="%"
              ></InputNumber>
            </Form.Item>
            <AlignmentBatchSizeInput disabled={initialIsLoading} />
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
              label={
                <Typography.Text>
                  Minimum Seconds
                  <HelpIcon
                    content={
                      <Typography>
                        Audio files with a duration less than minimum seconds
                        will not be preprocessed and ignored during training.
                      </Typography>
                    }
                    style={{ marginLeft: 8 }}
                  />
                </Typography.Text>
              }
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
              label={
                <Typography.Text>
                  Maximum Seconds
                  <HelpIcon
                    content={
                      <Typography>
                        Audio files with a duration longer than maximum seconds
                        will not be preprocessed and ignored during training.
                        Note that the length of the largest audio in the dataset
                        is proportional to the maximum amount of RAM/VRAM used
                        during training. If you encounter out of memory issues
                        reducing this value may be an option.
                      </Typography>
                    }
                    style={{ marginLeft: 8 }}
                  />
                </Typography.Text>
              }
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
                <Typography.Text>
                  Apply Audio Normalization
                  <HelpIcon
                    content={
                      <Typography>
                        Apply peak normalization where each audio is divided by
                        its highest absolute PCM value so the loudness in each
                        audio is roughly the same. Normalizing loudness can
                        improve training especially when the dataset comes from
                        multiple sources.
                      </Typography>
                    }
                    style={{ marginLeft: 8 }}
                  />
                </Typography.Text>
              </Checkbox>
            </Form.Item>
          </Collapse.Panel>
          <Collapse.Panel header="Acoustic Model" key="acoustic model">
            <LearningRateInput
              name="acousticLearningRate"
              disabled={initialIsLoading}
            />
            <TrainingStepsInput
              disabled={initialIsLoading}
              name="acousticTrainingIterations"
            />
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
            <LearningRateInput
              name="vocoderLearningRate"
              disabled={initialIsLoading}
            />
            <TrainingStepsInput
              disabled={initialIsLoading}
              name="vocoderTrainingIterations"
            />
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
