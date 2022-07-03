import React, { useState, useRef, useEffect, ReactElement } from "react";
import { useHistory } from "react-router-dom";
import { FormInstance } from "rc-field-form";
import { useDispatch } from "react-redux";
import { sampleSplittingRunInitialValues } from "../../../config";
import {
  SampleSplittingRunConfigInterface,
  SampleSplittingRunInterface,
} from "../../../interfaces";
import { notifySave } from "../../../utils";
import {
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL,
  FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES,
  UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL_TYPES,
} from "../../../channels";
import AlignmentBatchSizeInput from "../../../components/inputs/AlignmentBatchSizeInput";
import MaximumWorkersInput from "../../../components/inputs/MaximumWorkersInput";
import SkipOnErrorInput from "../../../components/inputs/SkipOnErrorInput";
import DeviceInput from "../../../components/inputs/DeviceInput";
import DatasetInput from "../../../components/inputs/DatasetInput";
import RunConfiguration from "../../../components/runs/RunConfiguration";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import { fetchNames } from "../PreprocessingRuns";
import { addToQueue } from "../../../features/runManagerSlice";
const { ipcRenderer } = window.require("electron");

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface;
}): ReactElement {
  const dispatch = useDispatch();

  const isMounted = useRef(false);
  const [initialIsLoading, setInitialIsLoading] = useState(true);
  const history = useHistory();
  const navigateNextRef = useRef<boolean>(false);
  const formRef =
    useRef<FormInstance<SampleSplittingRunConfigInterface> | null>();

  const onBack = () => {
    history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
  };

  const onNext = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const onDefaults = () => {
    formRef.current?.setFieldsValue({
      ...sampleSplittingRunInitialValues,
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
    const args: UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL_TYPES["IN"]["ARGS"] = {
      ...run,
      configuration: {
        ...sampleSplittingRunInitialValues,
        ...formRef.current?.getFieldsValue(),
      },
    };

    ipcRenderer
      .invoke(UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL.IN, args)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        if (navigateNextRef.current) {
          if (run.stage === "not_started") {
            dispatch(
              addToQueue({
                ID: run.ID,
                type: "sampleSplittingRun",
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
    const args: FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      ID: run.ID,
    };
    ipcRenderer
      .invoke(FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL.IN, args)
      .then((runs: FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        if (initialIsLoading) {
          setInitialIsLoading(false);
        }
        formRef.current?.setFieldsValue(runs[0].configuration);
      });
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
    <RunConfiguration
      title="Configuration"
      docsUrl="/usage/sample-splitting#configuration"
      forms={
        <>
          <DatasetInput
            disabled={initialIsLoading || hasStarted}
            docsUrl="/usage/sample-splitting#configuration"
          />
          <DeviceInput
            disabled={initialIsLoading}
            docsUrl="/usage/sample-splitting#configuration"
          />
          <MaximumWorkersInput
            disabled={initialIsLoading}
            docsUrl="/usage/sample-splitting#configuration"
          />
          <SkipOnErrorInput
            disabled={initialIsLoading}
            docsUrl="/usage/sample-splitting#configuration"
          />
          <AlignmentBatchSizeInput
            disabled={initialIsLoading}
            docsUrl="/usage/sample-splitting#configuration"
          />
        </>
      }
      hasStarted={run.stage !== "not_started"}
      isDisabled={initialIsLoading}
      onBack={onBack}
      onDefaults={onDefaults}
      onSave={onSave}
      onNext={onNext}
      formRef={formRef}
      initialValues={sampleSplittingRunInitialValues}
      onFinish={onFinish}
      fetchNames={() => fetchNames(run.ID)}
    />
  );
}
