import React, { useState, useEffect, useRef, ReactElement } from "react";
import { Table, Space, Input, Button, Typography, notification } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import type { FilterConfirmProps } from "antd/lib/table/interface";
import type { ColumnType } from "antd/lib/table";
import type { InputRef } from "antd";
import {
  FINISH_SAMPLE_SPLITTING_RUN_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
  FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL,
  UPDATE_SAMPLE_SPLITTING_SAMPLE_CHANNEL,
  GET_AUDIO_DATA_URL_CHANNEL,
} from "../../../channels";
import { useHistory } from "react-router-dom";
import AudioBottomBar from "../../../components/audio_player/AudioBottomBar";
import { defaultPageOptions } from "../../../config";
import {
  RunInterface,
  SampleSplittingRunInterface,
  SampleSplittingSampleInterface,
} from "../../../interfaces";
import RunCard from "../../../components/cards/RunCard";
import { PREPROCESSING_RUNS_ROUTE } from "../../../routes";
const { ipcRenderer } = window.require("electron");

export default function ChooseSamples({
  onStepChange,
  running,
  continueRun,
  run,
  stopRun,
}: {
  onStepChange: (current: number) => void;
  running: RunInterface | null;
  continueRun: (run: RunInterface) => void;
  run: SampleSplittingRunInterface | null;
  stopRun: () => void;
}): ReactElement {
  const isMounted = useRef(false);
  const history = useHistory();
  const [samples, setSamples] = useState<SampleSplittingSampleInterface[]>([]);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const playFuncRef = useRef<null | (() => void)>(null);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);

  const searchInput = useRef<InputRef>(null);

  const handleSearch = (
    selectedKeys: string[],
    confirm: (param?: FilterConfirmProps) => void,
    dataIndex: any
  ) => {
    confirm();
  };

  const handleReset = (clearFilters: () => void) => {
    clearFilters();
  };

  const onNewTextEdit = (
    record: SampleSplittingSampleInterface,
    newText: string
  ) => {
    ipcRenderer
      .invoke(UPDATE_SAMPLE_SPLITTING_SAMPLE_CHANNEL.IN, record.ID, newText)
      .then(fetchSamples);
  };

  const getColumnSearchProps = (dataIndex: any): ColumnType<any> => ({
    filterDropdown: ({
      setSelectedKeys,
      selectedKeys,
      confirm,
      clearFilters,
    }) => (
      <div style={{ padding: 8 }}>
        <Input
          ref={searchInput}
          placeholder={`Search ${dataIndex}`}
          value={selectedKeys[0]}
          onChange={(e) =>
            setSelectedKeys(e.target.value ? [e.target.value] : [])
          }
          onPressEnter={() =>
            handleSearch(selectedKeys as string[], confirm, dataIndex)
          }
          style={{ marginBottom: 8, display: "block" }}
        />
        <Space>
          <Button
            type="primary"
            onClick={() =>
              handleSearch(selectedKeys as string[], confirm, dataIndex)
            }
            icon={<SearchOutlined />}
            size="small"
            style={{ width: 90 }}
          >
            Search
          </Button>
          <Button
            onClick={() => {
              setSelectedKeys([]);
              handleReset(clearFilters);
              handleSearch([], confirm, dataIndex);
            }}
            size="small"
          >
            Reset
          </Button>
        </Space>
      </div>
    ),
    filterIcon: (filtered: boolean) => (
      <SearchOutlined style={{ color: filtered ? "#1890ff" : undefined }} />
    ),
    onFilter: (value, record) =>
      record[dataIndex]
        .toString()
        .toLowerCase()
        .includes((value as string).toLowerCase()),
    onFilterDropdownVisibleChange: (visible) => {
      if (visible) {
        setTimeout(() => searchInput.current?.select(), 100);
      }
    },
  });

  const removeSamples = (sampleIDs: number[]) => {
    if (run === null) {
      return;
    }

    ipcRenderer
      .invoke(REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL.IN, sampleIDs)
      .then(fetchSamples);
  };

  const onSamplesRemove = () => {
    if (run === null) {
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
    if (run === null) {
      return;
    }
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

  const onFinish = () => {
    if (run === null) {
      return;
    }
    ipcRenderer
      .invoke(FINISH_SAMPLE_SPLITTING_RUN_CHANNEL.IN, run.ID)
      .then(() => {
        ipcRenderer
          .invoke(REMOVE_PREPROCESSING_RUN_CHANNEL.IN, {
            ID: run.ID,
            type: "sampleSplittingRun",
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
    if (run === null) {
      return;
    }
    fetchSamples();
  }, []);

  const columns = [
    {
      title: "Old Text",
      dataIndex: "oldText",
      key: "oldText",
      ...getColumnSearchProps("oldText"),
    },
    {
      title: "New Text",
      dataIndex: "newText",
      key: "newText",
      ...getColumnSearchProps("newText"),
      render: (text: any, record: SampleSplittingSampleInterface) => (
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
    {
      title: "Reason",
      dataIndex: "reason",
      key: "reason",
      ...getColumnSearchProps("reason"),
    },
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
  ];

  return (
    <>
      <RunCard
        title={`The following samples will be split ... (${samples.length} total)`}
        buttons={[
          <Button onClick={onBackClick}>Back</Button>,
          <Button onClick={onFinish} type="primary">
            Apply Sample Splitting
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
