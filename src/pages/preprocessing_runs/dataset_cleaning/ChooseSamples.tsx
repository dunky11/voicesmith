import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Button, Table, Space, Typography, Slider, Form } from "antd";
import { FormInstance } from "rc-field-form";
import { useHistory } from "react-router-dom";
import { defaultPageOptions, cleaningRunInitialValues } from "../../../config";
import HelpIcon from "../../../components/help/HelpIcon";
import { numberCompare, stringCompare } from "../../../utils";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import {
  CleaningRunInterface,
  CleaningRunSampleInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import {
  FETCH_CLEANING_RUN_SAMPLES_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
  FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES,
  REMOVE_CLEANING_RUN_SAMPLES_CHANNEL,
  REMOVE_CLEANING_RUN_SAMPLES_CHANNEL_TYPES,
  FETCH_CLEANING_RUNS_CHANNEL,
} from "../../../channels";
const { ipcRenderer } = window.require("electron");

export default function ChooseSamples({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: CleaningRunInterface;
}): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [samples, setSamples] = useState<CleaningRunSampleInterface[]>([]);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);
  const formRef = useRef<FormInstance | null>();
  const navigateNextRef = useRef<boolean>(false);
  const [initialIsLoading, setInitialIsLoading] = useState(true);

  const removeSamples = (sampleIDs: number[]) => {
    const args: REMOVE_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["ARGS"] = {
      sampleIDs,
    };
    ipcRenderer
      .invoke(REMOVE_CLEANING_RUN_SAMPLES_CHANNEL.IN, args)
      .then(fetchSamples);
  };

  const onSamplesRemove = () => {
    removeSamples(
      selectedRowKeys.map((selectedRowKey: string) => parseInt(selectedRowKey))
    );
  };

  const onBackClick = () => {
    onStepChange(1);
  };

  const onNext = () => {
    if (formRef.current === null) {
      return;
    }
    navigateNextRef.current = true;
    formRef.current.submit();
  };

  const fetchSamples = () => {
    const args: FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["ARGS"] = {
      runID: run.ID,
    };
    ipcRenderer
      .invoke(FETCH_CLEANING_RUN_SAMPLES_CHANNEL.IN, args)
      .then(
        (samples: FETCH_CLEANING_RUN_SAMPLES_CHANNEL_TYPES["IN"]["OUT"]) => {
          setSamples(samples);
        }
      );
  };

  const loadAudio = (filePath: string) => {
    ipcRenderer
      .invoke("get-audio-data-url", filePath)
      .then((dataUrl: string) => {
        setAudioDataURL(dataUrl);
        if (playFuncRef.current != null) {
          playFuncRef.current();
        }
      });
  };

  const onFinish = () => {
    ipcRenderer.removeAllListeners("finish-cleaning-run-reply");
    ipcRenderer.on(
      "finish-cleaning-run-reply",
      (
        _: any,
        message: {
          type: string;
          progress?: number;
        }
      ) => {
        switch (message.type) {
          case "progress": {
            break;
          }
          case "finished": {
            ipcRenderer
              .invoke(REMOVE_PREPROCESSING_RUN_CHANNEL.IN, {
                ID: run.ID,
                type: "dSCleaning",
              })
              .then(() => {
                history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
              });
            break;
          }
          default: {
            throw new Error(
              `No branch selected in switch-statement, '${message.type}' is not a valid case ...`
            );
          }
        }
      }
    );
    ipcRenderer.send("finish-cleaning-run", run.ID);
  };

  const fetchConfiguration = () => {
    ipcRenderer
      .invoke(FETCH_CLEANING_RUNS_CHANNEL.IN, run.ID)
      .then((runs: CleaningRunInterface[]) => {
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
      ipcRenderer.removeAllListeners("finish-cleaning-run-reply");
    };
  });

  useEffect(() => {
    fetchSamples();
    fetchConfiguration();
  }, []);

  const columns = [
    {
      title: "Text",
      dataIndex: "text",
      key: "text",
      sorter: {
        compare: (
          a: CleaningRunSampleInterface,
          b: CleaningRunSampleInterface
        ) => stringCompare(a.text, b.text),
      },
    },
    {
      title: "Sample Quality",
      dataIndex: "qualityScore",
      key: "qualityScore",
      sorter: {
        compare: (
          a: CleaningRunSampleInterface,
          b: CleaningRunSampleInterface
        ) => numberCompare(a.qualityScore, b.qualityScore),
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: CleaningRunSampleInterface) => (
        <Space size="middle">
          <a
            onClick={() => {
              loadAudio(record.audioPath);
            }}
          >
            Play
          </a>
        </Space>
      ),
    },
  ];

  return (
    <>
      <RunCard
        title={`Choose Samples (${samples.length} total)`}
        buttons={[
          <Button onClick={onBackClick}>Back</Button>,
          <Button onClick={onNext} disabled={initialIsLoading} type="primary">
            Remove Samples From Dataset
          </Button>,
        ]}
      >
        <div style={{ width: "100%" }}>
          <Form
            layout="vertical"
            ref={(node: any) => {
              formRef.current = node;
            }}
            initialValues={cleaningRunInitialValues}
            onFinish={onFinish}
          >
            <Form.Item
              label={
                <Typography.Text>
                  Remove all samples with a sample quality of less than.
                  <HelpIcon docsUrl={"TODO"} style={{ marginLeft: 8 }} />
                </Typography.Text>
              }
              name="removeIfQualityLessThan"
            >
              <Slider min={0} max={1} disabled={initialIsLoading} />
            </Form.Item>
          </Form>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              disabled={selectedRowKeys.length === 0}
              onClick={onSamplesRemove}
            >
              Do not remove selected
            </Button>
          </div>
          <Table
            size="small"
            pagination={defaultPageOptions}
            bordered
            style={{ width: "100%" }}
            columns={columns}
            rowSelection={{
              selectedRowKeys: selectedRowKeys,
              onChange: (selectedRowKeys: any[]) => {
                setSelectedRowKeys(selectedRowKeys);
              },
            }}
            dataSource={samples.map((sample: CleaningRunSampleInterface) => ({
              ...sample,
              key: sample.ID,
            }))}
          ></Table>
        </div>
      </RunCard>
      <div
        style={{
          display: audioDataURL === null ? "none" : "block",
          marginTop: 64,
        }}
      >
        <AudioBottomBar
          src={audioDataURL}
          playFuncRef={playFuncRef}
        ></AudioBottomBar>
      </div>
    </>
  );
}
