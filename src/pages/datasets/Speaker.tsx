import React, { useState, useEffect, useRef, ReactElement } from "react";
import {
  Card,
  Button,
  Table,
  Breadcrumb,
  Space,
  Typography,
  InputRef,
} from "antd";
import { Link } from "react-router-dom";
import { defaultPageOptions } from "../../config";
import AudioBottomBar from "../../components/audio_player/AudioBottomBar";
import { stringCompare, getSearchableColumn } from "../../utils";
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
  GET_AUDIO_DATA_URL_CHANNEL,
} from "../../channels";
import { DATASETS_ROUTE } from "../../routes";
import { useDispatch } from "react-redux";
import { setNavIsDisabled } from "../../features/navigationSettingsSlice";
const { ipcRenderer } = window.require("electron");

export default function Speaker({
  speaker,
  setSelectedSpeakerID,
  fetchDataset,
  datasetID,
  datasetName,
  isDisabled,
}: {
  speaker: SpeakerInterface | null;
  setSelectedSpeakerID: (ID: number | null) => void;
  fetchDataset: () => void;
  datasetID: number | null;
  datasetName: string | null;
  isDisabled: boolean;
}): ReactElement {
  const dispatch = useDispatch();
  const [isLoading, setIsLoading] = useState(false);
  const isMounted = useRef(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const onSpeakerBackClick = () => {
    setSelectedSpeakerID(null);
  };
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);
  const searchInput = useRef<InputRef>(null);

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
        dispatch(setNavIsDisabled(false));
        if (!isMounted.current) {
          return;
        }
        setIsLoading(false);
        fetchDataset();
      });
    setIsLoading(true);
    dispatch(setNavIsDisabled(true));
  };

  const onFilesAddClick = () => {
    ipcRenderer
      .invoke(PICK_SPEAKER_FILES_CHANNEL.IN)
      .then((filePaths: FileInterface[]) => {
        setIsLoading(false);
        dispatch(setNavIsDisabled(false));
        if (speaker === null || !isMounted.current || filePaths.length === 0) {
          return;
        }
        addSamplesToSpeaker(filePaths);
      });
    setIsLoading(true);
    dispatch(setNavIsDisabled(true));
  };

  const addSamplesToSpeaker = (filePaths: FileInterface[]) => {
    if (speaker === null || datasetID === null) {
      return;
    }
    ipcRenderer
      .invoke(ADD_SAMPLES_CHANNEL.IN, speaker, filePaths, datasetID)
      .then(() => {
        setIsLoading(false);
        dispatch(setNavIsDisabled(false));
        fetchDataset();
      });
    setIsLoading(true);
    dispatch(setNavIsDisabled(true));
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
      .invoke(GET_AUDIO_DATA_URL_CHANNEL.IN, filePath)
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
    getSearchableColumn(
      {
        title: "Text Path",
        dataIndex: "txtPath",
        key: "txtPath",
        sorter: {
          compare: (a: SpeakerSampleInterface, b: SpeakerSampleInterface) =>
            stringCompare(a.txtPath, b.txtPath),
        },
      },
      "txtPath",
      searchInput
    ),
    getSearchableColumn(
      {
        title: "Audio Path",
        dataIndex: "audioPath",
        key: "audioPath",
        sorter: {
          compare: (a: SpeakerSampleInterface, b: SpeakerSampleInterface) =>
            stringCompare(a.audioPath, b.audioPath),
        },
      },
      "audioPath",
      searchInput
    ),
    getSearchableColumn(
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
            editable={
              isLoading
                ? null
                : {
                    onChange: (newText) => {
                      onTextChange(record.ID, newText);
                    },
                  }
            }
          >
            {record.text}
          </Typography.Text>
        ),
      },
      "text",
      searchInput
    ),
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
          <Link to={DATASETS_ROUTE.SELECTION.ROUTE}>Datasets</Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item onClick={onSpeakerBackClick}>
          <Link to={DATASETS_ROUTE.EDIT.ROUTE}>{datasetName}</Link>
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
              disabled={isDisabled || isLoading}
              style={{ marginRight: 8 }}
            >
              Add Files
            </Button>
            <Button
              disabled={selectedRowKeys.length === 0 || isDisabled || isLoading}
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
            rowSelection={
              isDisabled || isLoading
                ? null
                : {
                    selectedRowKeys: selectedRowKeys,
                    onChange: (selectedRowKeys: any[]) => {
                      setSelectedRowKeys(selectedRowKeys);
                    },
                  }
            }
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
