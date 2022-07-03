import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Form, Collapse, InputNumber, Checkbox, Typography } from "antd";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch } from "react-redux";
import {
  TrainingRunConfigInterface,
  TrainingRunInterface,
} from "../../interfaces";
import { trainingRunInitialValues } from "../../config";
import { notifySave } from "../../utils";
import RunConfiguration from "../../components/runs/RunConfiguration";
import TrainingStepsInput from "../../components/inputs/TrainingStepsInput";
import LearningRateInput from "../../components/inputs/LearningRateInput";
import SkipOnErrorInput from "../../components/inputs/SkipOnErrorInput";
import DeviceInput from "../../components/inputs/DeviceInput";
import DatasetInput from "../../components/inputs/DatasetInput";
import MaximumWorkersInput from "../../components/inputs/MaximumWorkersInput";
import AlignmentBatchSizeInput from "../../components/inputs/AlignmentBatchSizeInput";
import BatchSizeInput from "../../components/inputs/BatchSizeInput";
import GradientAccumulationStepsInput from "../../components/inputs/GradientAccumulationStepsInput";
import RunValidationEveryInput from "../../components/inputs/RunValidationEveryInput";
import TrainOnlySpeakerEmbedsUntilInput from "../../components/inputs/TrainOnlySpeakerEmbedsUntilInput";
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
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const [initialIsLoading, setInitialIsLoading] = useState(true);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef = useRef<FormInstance | null>();

  const onBack = () => {
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

  const onNext = () => {
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
    <RunConfiguration
      title="Configure the Training Run"
      hasStarted={hasStarted}
      isDisabled={initialIsLoading}
      onBack={onBack}
      onDefaults={onDefaults}
      onSave={onSave}
      onNext={onNext}
      formRef={formRef}
      initialValues={trainingRunInitialValues}
      onFinish={onFinish}
      fetchNames={fetchNames}
      docsUrl="/usage/training"
      forms={
        <>
          <DatasetInput
            disabled={initialIsLoading || hasStarted}
            docsUrl="/usage/training#configuration"
          />
          <DeviceInput
            disabled={initialIsLoading}
            docsUrl="/usage/training#configuration"
          />
          <MaximumWorkersInput
            disabled={initialIsLoading}
            docsUrl="/usage/training#configuration"
          />
          <SkipOnErrorInput
            disabled={initialIsLoading}
            docsUrl="/usage/training#configuration"
          />
          <Collapse style={{ width: "100%" }}>
            <Collapse.Panel header="Preprocessing" key="preprocessing">
              <Form.Item
                label={
                  <Typography.Text>
                    Validation Size
                    <HelpIcon
                      docsUrl="/usage/training#preprocessing-configuration"
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
              <AlignmentBatchSizeInput
                docsUrl="/usage/training#preprocessing-configuration"
                disabled={initialIsLoading}
              />
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
                      docsUrl="/usage/training#preprocessing-configuration"
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
                      docsUrl="/usage/training#preprocessing-configuration"
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
                      docsUrl="/usage/training#preprocessing-configuration"
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
                docsUrl="/usage/training#acoustic-model-configuration"
              />
              <TrainingStepsInput
                name="acousticTrainingIterations"
                disabled={initialIsLoading}
                docsUrl="/usage/training#acoustic-model-configuration"
              />
              <BatchSizeInput
                name="acousticBatchSize"
                disabled={initialIsLoading}
                docsUrl="/usage/training#acoustic-model-configuration"
              />
              <GradientAccumulationStepsInput
                name="acousticGradAccumSteps"
                disabled={initialIsLoading}
                docsUrl="/usage/training#acoustic-model-configuration"
              />
              <RunValidationEveryInput
                name="acousticValidateEvery"
                disabled={initialIsLoading}
                docsUrl="/usage/training#acoustic-model-configuration"
              />
              <TrainOnlySpeakerEmbedsUntilInput
                rules={[
                  ({
                    getFieldValue,
                  }: {
                    getFieldValue: (name: any) => any;
                  }) => ({
                    validator(_: any, value: any) {
                      if (value > getFieldValue("acousticTrainingIterations")) {
                        return Promise.reject(
                          new Error("Cannot be smaller than training steps")
                        );
                      }
                      return Promise.resolve();
                    },
                  }),
                ]}
                name="onlyTrainSpeakerEmbUntil"
                docsUrl="/usage/training#acoustic-model-configuration"
              />
            </Collapse.Panel>
            <Collapse.Panel header="Vocoder" key="vocoder">
              <LearningRateInput
                name="vocoderLearningRate"
                disabled={initialIsLoading}
                docsUrl="/usage/training#vocoder-model-configuration"
              />
              <TrainingStepsInput
                disabled={initialIsLoading}
                name="vocoderTrainingIterations"
                docsUrl="/usage/training#vocoder-model-configuration"
              />
              <BatchSizeInput
                name="vocoderBatchSize"
                disabled={initialIsLoading}
                docsUrl="/usage/training#vocoder-model-configuration"
              />
              <GradientAccumulationStepsInput
                name="vocoderGradAccumSteps"
                disabled={initialIsLoading}
                docsUrl="/usage/training#vocoder-model-configuration"
              />
              <RunValidationEveryInput
                name="vocoderValidateEvery"
                disabled={initialIsLoading}
                docsUrl="/usage/training#vocoder-model-configuration"
              />
            </Collapse.Panel>
          </Collapse>
        </>
      }
    />
  );
}
