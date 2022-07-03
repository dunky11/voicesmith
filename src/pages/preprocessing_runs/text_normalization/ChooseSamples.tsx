import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Table, Space, Input, Button, Typography, notification } from "antd";
import type { InputRef } from "antd";
import { getSearchableColumn } from "../../../utils";
import {
  FINISH_TEXT_NORMALIZATION_RUN_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
  FETCH_TEXT_NORMALIZATION_SAMPLES_CHANNEL,
  REMOVE_TEXT_NORMALIZATION_SAMPLES_CHANNEL,
  EDIT_TEXT_NORMALIZATION_SAMPLE_NEW_TEXT_CHANNEL,
  GET_AUDIO_DATA_URL_CHANNEL,
} from "../../../channels";
import { useHistory } from "react-router-dom";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import { defaultPageOptions } from "../../../config";
import {
  TextNormalizationRunInterface,
  TextNormalizationSampleInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
const { ipcRenderer } = window.require("electron");

export default function ChooseSamples({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
}): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [samples, setSamples] = useState<TextNormalizationSampleInterface[]>(
    []
  );
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);

  const searchInput = useRef<InputRef>(null);

  const onNewTextEdit = (
    record: TextNormalizationSampleInterface,
    newText: string
  ) => {
    ipcRenderer
      .invoke(
        EDIT_TEXT_NORMALIZATION_SAMPLE_NEW_TEXT_CHANNEL.IN,
        record.ID,
        newText
      )
      .then(fetchSamples);
  };

  const removeSamples = (sampleIDs: number[]) => {
    ipcRenderer
      .invoke(REMOVE_TEXT_NORMALIZATION_SAMPLES_CHANNEL.IN, sampleIDs)
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

  const fetchSamples = () => {
    ipcRenderer
      .invoke(FETCH_TEXT_NORMALIZATION_SAMPLES_CHANNEL.IN, run.ID)
      .then((samples: TextNormalizationSampleInterface[]) => {
        setSamples(samples);
      });
  };

  const loadAudio = (filePath: string) => {
    ipcRenderer
      .invoke(GET_AUDIO_DATA_URL_CHANNEL.IN, filePath)
      .then((dataUrl: string) => {
        setAudioDataURL(dataUrl);
        if (playFuncRef.current != null) {
          playFuncRef.current();
        }
      });
  };

  const onFinish = () => {
    ipcRenderer
      .invoke(FINISH_TEXT_NORMALIZATION_RUN_CHANNEL.IN, run.ID)
      .then(() => {
        ipcRenderer
          .invoke(REMOVE_PREPROCESSING_RUN_CHANNEL.IN, {
            ID: run.ID,
            type: "textNormalizationRun",
          })
          .then(() => {
            notification["success"]({
              message: "Your dataset has been normalized",
              placement: "top",
            });
            history.push(PREPROCESSING_RUNS_ROUTE.RUN_SELECTION.ROUTE);
          });
      });
  };

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  });

  useEffect(() => {
    fetchSamples();
  }, []);

  const columns = [
    getSearchableColumn(
      { title: "Old Text", dataIndex: "oldText", key: "oldText" },
      "oldText",
      searchInput
    ),
    getSearchableColumn(
      {
        title: "New Text",
        dataIndex: "newText",
        key: "newText",
        render: (text: any, record: TextNormalizationSampleInterface) => (
          <Typography.Text
            editable={{
              tooltip: false,
              onChange: (newName: string) => {
                onNewTextEdit(record, newName);
              },
            }}
          >
            {record.newText}
          </Typography.Text>
        ),
      },
      "newText",
      searchInput
    ),
    getSearchableColumn(
      {
        title: "Reason",
        dataIndex: "reason",
        key: "reason",
      },
      "reason",
      searchInput
    ),
    {
      title: "",
      key: "action",
      render: (text: any, record: TextNormalizationSampleInterface) => (
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
        docsUrl="/usage/text-normalization#choose-samples"
        buttons={[
          <Button onClick={onBackClick}>Back</Button>,
          <Button onClick={onFinish} type="primary">
            Apply Text Normalization
          </Button>,
        ]}
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              disabled={selectedRowKeys.length === 0}
              onClick={onSamplesRemove}
            >
              Do not normalize selected
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
            dataSource={samples.map(
              (sample: TextNormalizationSampleInterface) => ({
                ...sample,
                key: sample.ID,
              })
            )}
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
