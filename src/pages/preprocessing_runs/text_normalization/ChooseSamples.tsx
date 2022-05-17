import React, { useState, useEffect, useRef } from "react";
import { Card, Button, Table, Space } from "antd";
import { useHistory } from "react-router-dom";
import { numberCompare, stringCompare } from "../../../utils";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import { defaultPageOptions } from "../../../config";
import {
  NoisySampleInterface,
  RunInterface,
  TextNormalizationSampleInterface,
} from "../../../interfaces";
const { ipcRenderer } = window.require("electron");

export default function ChooseSamples({
  onStepChange,
  selectedID,
  running,
  continueRun,
  stage,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  selectedID: number | null;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  stage:
    | "not_started"
    | "text_normalization"
    | "choose_samples"
    | "finished"
    | null;
  stopRun: () => void;
}) {
  const isMounted = useRef(false);
  const history = useHistory();
  const [samples, setSamples] = useState<TextNormalizationSampleInterface[]>(
    []
  );
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);

  const removeSamples = (sampleIDs: number[]) => {
    if (selectedID === null) {
      return;
    }

    ipcRenderer
      .invoke("remove-text-normalization-samples", sampleIDs)
      .then(fetchSamples);
  };

  const onSamplesRemove = () => {
    if (selectedID === null) {
      return;
    }
    removeSamples(
      selectedRowKeys.map((selectedRowKey: string) => parseInt(selectedRowKey))
    );
  };

  const onBackClick = () => {
    onStepChange(1);
  };

  const fetchSamples = () => {
    if (selectedID === null) {
      return;
    }
    ipcRenderer
      .invoke("fetch-text-normalization-samples", selectedID)
      .then((samples: TextNormalizationSampleInterface[]) => {
        setSamples(samples);
      });
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
    if (selectedID === null) {
      return;
    }
    ipcRenderer.removeAllListeners("finish-text-normalization-run-reply");
    ipcRenderer.on(
      "finish-text-normalization-run-reply",
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
              .invoke("remove-preprocessing-run", {
                ID: selectedID,
                type: "textNormalization",
              })
              .then(() => {
                history.push("/preprocessing-runs/run-selection");
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
    ipcRenderer.send("finish-text-normalization-run", selectedID);
  };

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners("finish-text-normalization-run-reply");
    };
  });

  useEffect(() => {
    if (selectedID === null) {
      return;
    }
    fetchSamples();
  }, []);

  const columns = [
    {
      title: "Old Text",
      dataIndex: "oldText",
      key: "oldText",
    },
    {
      title: "New Text",
      dataIndex: "newText",
      key: "newText",
    },
    {
      title: "Reason",
      dataIndex: "reason",
      key: "reason",
    },
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
      <Card
        title={`The following samples will be normalized ... (${samples.length} total)`}
        actions={[
          <div
            key="next-button-wrapper"
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginRight: 24,
            }}
          >
            <Button onClick={onBackClick} style={{ marginRight: 8 }}>
              Back
            </Button>
            <Button onClick={onFinish} type="primary">
              Apply Text Normalization
            </Button>
          </div>,
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
