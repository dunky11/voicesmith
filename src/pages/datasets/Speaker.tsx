import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Card, Button, Table, Breadcrumb, Space, Typography } from "antd";
import { Link } from "react-router-dom";
import { defaultPageOptions } from "../../config";
import AudioBottomBar from "../../components/audio_player/AudioBottomBar";
import { stringCompare } from "../../utils";
import {
  SpeakerInterface,
  SpeakerSampleInterface,
  FileInterface,
} from "../../interfaces";
import {
  REMOVE_SAMPLES_CHANNEL,
  PICK_SPEAKER_FILES_CHANNEL,
  ADD_SAMPLES_CHANNEL,
  EDIT_SAMPLE_TEXT_CHANNEL,
} from "../../channels";
const { ipcRenderer } = window.require("electron");

export default function Speaker({
  speaker,
  setSelectedSpeakerID,
  fetchDataset,
  datasetID,
  datasetName,
  referencedBy,
}: {
  speaker: SpeakerInterface | null;
  setSelectedSpeakerID: (ID: number | null) => void;
  fetchDataset: () => void;
  datasetID: number | null;
  datasetName: string | null;
  referencedBy: string | null;
}): ReactElement {
  const isMounted = useRef(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const onSpeakerBackClick = () => {
    setSelectedSpeakerID(null);
  };
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);

  const removeSamples = (sampleIDs: number[]) => {
    if (datasetID === null || speaker === null) {
      return;
    }
    const samples = speaker.samples.filter((sample: SpeakerSampleInterface) => {
      if (sample.ID === undefined) {
        throw Error("Sample ID cannot be undefined ...");
      }
      return sampleIDs.includes(sample.ID);
    });
    ipcRenderer
      .invoke(REMOVE_SAMPLES_CHANNEL.IN, datasetID, speaker.ID, samples)
      .then(() => {
        if (!isMounted.current) {
          return;
        }
        fetchDataset();
      });
  };

  const onFilesAddClick = () => {
    ipcRenderer
      .invoke(PICK_SPEAKER_FILES_CHANNEL.IN)
      .then((filePaths: FileInterface[]) => {
        if (speaker === null || !isMounted.current || filePaths.length === 0) {
          return;
        }
        addSamplesToSpeaker(filePaths);
      });
  };

  const addSamplesToSpeaker = (filePaths: FileInterface[]) => {
    if (speaker === null || datasetID === null) {
      return;
    }
    ipcRenderer
      .invoke(ADD_SAMPLES_CHANNEL.IN, speaker, filePaths, datasetID)
      .then(fetchDataset);
  };

  const onSamplesRemove = () => {
    if (speaker === null) {
      return;
    }
    removeSamples(
      selectedRowKeys.map((selectedRowKey: string) => parseInt(selectedRowKey))
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

  const onTextChange = (sampleID: number, newText: string) => {
    ipcRenderer
      .invoke(EDIT_SAMPLE_TEXT_CHANNEL.IN, sampleID, newText.trim())
      .then(fetchDataset);
  };

  const columns = [
    {
      title: "Text Path",
      dataIndex: "txtPath",
      key: "txtPath",
      sorter: {
        compare: (a: SpeakerSampleInterface, b: SpeakerSampleInterface) =>
          stringCompare(a.txtPath, b.txtPath),
      },
    },
    {
      title: "Audio Path",
      dataIndex: "audioPath",
      key: "audioPath",
      sorter: {
        compare: (a: SpeakerSampleInterface, b: SpeakerSampleInterface) =>
          stringCompare(a.audioPath, b.audioPath),
      },
    },
    {
      title: "Text",
      dataIndex: "text",
      key: "text",
      sorter: {
        compare: (a: SpeakerSampleInterface, b: SpeakerSampleInterface) =>
          stringCompare(a.text, b.text),
      },
      render: (text: any, record: SpeakerSampleInterface) => (
        <Typography.Text
          editable={{
            onChange: (newText) => {
              onTextChange(record.ID, newText);
            },
          }}
        >
          {record.text}
        </Typography.Text>
      ),
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: SpeakerSampleInterface) => (
        <Space size="middle">
          <a
            onClick={() => {
              loadAudio(record.fullAudioPath);
            }}
          >
            Play
          </a>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  });

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>
          <Link to="/datasets/dataset-selection">Datasets</Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item onClick={onSpeakerBackClick}>
          <Link to="/datasets/dataset-edit">{datasetName}</Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>{speaker?.name}</Breadcrumb.Item>
      </Breadcrumb>
      <Card
        title={`Samples of speaker '${speaker?.name}'`}
        actions={[
          <div
            key="next-button-wrapper"
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginRight: 24,
            }}
          >
            <Button onClick={onSpeakerBackClick}>Back</Button>
          </div>,
        ]}
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              onClick={onFilesAddClick}
              disabled={referencedBy !== null}
              style={{ marginRight: 8 }}
            >
              Add Files
            </Button>
            <Button
              disabled={selectedRowKeys.length === 0 || referencedBy !== null}
              onClick={onSamplesRemove}
            >
              Remove Selected
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
            dataSource={speaker?.samples.map(
              (sample: SpeakerSampleInterface) => ({
                ...sample,
                key: sample.ID,
              })
            )}
          ></Table>
        </div>
      </Card>
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
