import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Table, Space, Button, List, Typography } from "antd";
import { useDispatch, useSelector } from "react-redux";
import type { InputRef } from "antd";
import { getWouldContinueRun, getSearchableColumn } from "../../../utils";
import { RootState } from "../../../app/store";
import {
  FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL,
  GET_AUDIO_DATA_URL_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_RUN_STAGE_CHANNEL,
} from "../../../channels";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import { defaultPageOptions } from "../../../config";
import {
  RunInterface,
  SampleSplittingRunInterface,
  SampleSplittingSampleInterface,
  SampleSplittingSplitInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { addToQueue } from "../../../features/runManagerSlice";

const { ipcRenderer } = window.require("electron");

export default function ChooseSamples({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface;
}): ReactElement {
  const dispatch = useDispatch();
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const isMounted = useRef(false);
  const [samples, setSamples] = useState<SampleSplittingSampleInterface[]>([]);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);
  const searchInput = useRef<InputRef>(null);

  const wouldContinueRun = getWouldContinueRun(
    ["choose_samples", "apply_changes"],
    run.stage,
    running,
    "sampleSplittingRun",
    run.ID
  );

  const removeSamples = (sampleIDs: number[]) => {
    ipcRenderer
      .invoke(REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL.IN, sampleIDs)
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
      .invoke(FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL.IN, run.ID)
      .then((samples: SampleSplittingSampleInterface[]) => {
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

  const removeSampleSplit = (sampleID: number, sampleSplitIDs: number[]) => {
    ipcRenderer
      .invoke(
        REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL.IN,
        sampleID,
        sampleSplitIDs
      )
      .then(fetchSamples);
  };

  const onNext = () => {
    const navigateNext = () => {
      if (wouldContinueRun) {
        dispatch(
          addToQueue({ ID: run.ID, type: "sampleSplittingRun", name: run.name })
        );
      }
      onStepChange(3);
    };
    if (wouldContinueRun) {
      if (run.stage === "choose_samples") {
        ipcRenderer
          .invoke(
            UPDATE_SAMPLE_SPLITTING_RUN_STAGE_CHANNEL.IN,
            run.ID,
            "apply_changes"
          )
          .then(navigateNext);
      } else {
        navigateNext();
      }
    } else {
      navigateNext();
    }
  };

  const getNextButtonText = () => {
    if (wouldContinueRun) {
      return "Apply Changes";
    }
    return "Next";
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
      {
        title: "Text",
        dataIndex: "text",
        key: "text",
      },
      "text",
      searchInput
    ),
    getSearchableColumn(
      {
        title: "Speaker",
        dataIndex: "speakerName",
        key: "speakerName",
      },
      "speakerName",
      searchInput
    ),
    {
      title: "",
      key: "action",
      render: (text: any, record: SampleSplittingSampleInterface) => (
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
    {
      title: "Split Into",
      key: "action",
      render: (text: any, record: SampleSplittingSampleInterface) => (
        <List
          size="small"
          dataSource={record.splits}
          renderItem={(item: SampleSplittingSplitInterface, index: number) => (
            <List.Item>
              <Typography.Text>{`${index + 1}. ${item.text}`}</Typography.Text>
              <div>
                <a
                  onClick={() => {
                    loadAudio(item.audioPath);
                  }}
                  style={{ marginRight: 8 }}
                >
                  Play
                </a>
                <a
                  onClick={() => {
                    removeSampleSplit(record.ID, [item.ID]);
                  }}
                >
                  Delete
                </a>
              </div>
            </List.Item>
          )}
        />
      ),
    },
  ];

  return (
    <>
      <RunCard
        title={`The following samples will be split ... (${samples.length} total)`}
        buttons={[
          <Button onClick={onBackClick}>Back</Button>,
          <Button onClick={onNext} type="primary">
            {getNextButtonText()}
          </Button>,
        ]}
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              disabled={selectedRowKeys.length === 0}
              onClick={onSamplesRemove}
            >
              Do not split selected
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
              (sample: SampleSplittingSampleInterface) => ({
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
