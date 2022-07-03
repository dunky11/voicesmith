import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Button, Table, Space } from "antd";
import { useHistory } from "react-router-dom";
import { defaultPageOptions } from "../../../config";
import { numberCompare, stringCompare } from "../../../utils";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import {
  CleaningRunInterface,
  NoisySampleInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
import { REMOVE_PREPROCESSING_RUN_CHANNEL } from "../../../channels";
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
  const [noisySamples, setNoisySamples] = useState<NoisySampleInterface[]>([]);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);

  const removeSamples = (sampleIDs: number[]) => {
    ipcRenderer.invoke("remove-noisy-samples", sampleIDs).then(fetchSamples);
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
      .invoke("fetch-noisy-samples", run.ID)
      .then((noisySamples: NoisySampleInterface[]) => {
        setNoisySamples(noisySamples);
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

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners("finish-cleaning-run-reply");
    };
  });

  useEffect(() => {
    fetchSamples();
  }, []);

  const columns = [
    {
      title: "Text",
      dataIndex: "text",
      key: "text",
      sorter: {
        compare: (a: NoisySampleInterface, b: NoisySampleInterface) =>
          stringCompare(a.text, b.text),
      },
    },
    {
      title: "Sample Quality",
      dataIndex: "labelQuality",
      key: "labelQuality",
      sorter: {
        compare: (a: NoisySampleInterface, b: NoisySampleInterface) =>
          numberCompare(a.labelQuality, b.labelQuality),
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: NoisySampleInterface) => (
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
        title={`Choose Samples (${noisySamples.length} total)`}
        buttons={[
          <Button onClick={onBackClick}>Back</Button>,
          <Button onClick={onFinish} type="primary">
            Remove Samples From Dataset
          </Button>,
        ]}
      >
        <div style={{ width: "100%" }}>
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
            dataSource={noisySamples.map((sample: NoisySampleInterface) => ({
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
