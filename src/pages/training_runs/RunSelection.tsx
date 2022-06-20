import React, { useEffect, useState, useRef, ReactElement } from "react";
import {
  Table,
  Button,
  Card,
  Space,
  Tag,
  Breadcrumb,
  Popconfirm,
  Typography,
} from "antd";
import { SyncOutlined } from "@ant-design/icons";
import { useSelector, useDispatch } from "react-redux";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { POLL_LOGFILE_INTERVALL, defaultPageOptions } from "../../config";
import { useInterval, stringCompare } from "../../utils";
import {
  FETCH_TRAINING_RUNS_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL_TYPES,
  CREATE_TRAINING_RUN_CHANNEL,
} from "../../channels";
import { RootState } from "../../app/store";
import { setIsRunning, addToQueue } from "../../features/runManagerSlice";
const { ipcRenderer } = window.require("electron");

export default function RunSelection({
  removeTrainingRun,
  selectTrainingRun,
}: {
  removeTrainingRun: (run: TrainingRunInterface) => void;
  selectTrainingRun: (run: TrainingRunInterface) => void;
}): ReactElement {
  const running: RunInterface = useSelector((state: RootState) => {
    if (!state.runManager.isRunning || state.runManager.queue.length === 0) {
      return null;
    }
    return state.runManager.queue[0];
  });
  const runManager = useSelector((state: RootState) => state.runManager);
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRunInterface[]>([]);

  const getFirstPossibleName = () => {
    const names = trainingRuns.map(
      (preprocessingRun: TrainingRunInterface) =>
        preprocessingRun.configuration.name
    );
    let i = 1;
    let name = `Training Run ${i}`;
    while (names.includes(name)) {
      i += 1;
      name = `Training Run ${i}`;
    }
    return name;
  };

  const pollTrainingRuns = () => {
    const args: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"] = {
      withStatistics: false,
      ID: null,
    };
    ipcRenderer
      .invoke(FETCH_TRAINING_RUNS_CHANNEL.IN, args)
      .then((trainingRuns: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        setTrainingRuns(trainingRuns);
      });
  };

  const createRun = () => {
    const name = getFirstPossibleName();
    ipcRenderer
      .invoke(CREATE_TRAINING_RUN_CHANNEL.IN, name)
      .then(pollTrainingRuns);
  };

  const columns = [
    {
      title: "Name",
      key: "name",
      sorter: {
        compare: (a: TrainingRunInterface, b: TrainingRunInterface) =>
          stringCompare(a.configuration.name, b.configuration.name),
      },
      render: (text: any, record: TrainingRunInterface) => (
        <Typography.Text>{record.configuration.name}</Typography.Text>
      ),
    },
    {
      title: "Stage",
      dataIndex: "stage",
      key: "stage",
      sorter: {
        compare: (a: TrainingRunInterface, b: TrainingRunInterface) =>
          stringCompare(a.stage, b.stage),
      },
    },

    {
      title: "State",
      key: "action",
      sorter: {
        compare: (a: TrainingRunInterface, b: TrainingRunInterface) =>
          stringCompare(
            running !== null &&
              running.type === "trainingRun" &&
              a.ID === running.ID
              ? "running"
              : "not_running",
            running !== null &&
              running.type === "trainingRun" &&
              b.ID === running.ID
              ? "running"
              : "not_running"
          ),
      },
      render: (text: any, record: TrainingRunInterface) =>
        running !== null &&
        running.type === "trainingRun" &&
        record.ID === running.ID ? (
          <Tag icon={<SyncOutlined spin />} color="green">
            Running
          </Tag>
        ) : (
          <Tag color="orange">Not Running</Tag>
        ),
    },
    {
      title: "Dataset",
      key: "datasetName",
      sorter: {
        compare: (a: TrainingRunInterface, b: TrainingRunInterface) =>
          stringCompare(
            a.configuration.datasetName,
            b.configuration.datasetName
          ),
      },
      render: (text: any, record: TrainingRunInterface) => {
        return (
          <Typography.Text>{record.configuration.datasetName}</Typography.Text>
        );
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: TrainingRunInterface) => {
        const isRunning =
          running !== null &&
          running.type === "trainingRun" &&
          record.ID === running.ID;

        const getRunningLink = () => {
          if (isRunning) {
            return (
              <a
                onClick={() => {
                  dispatch(setIsRunning(false));
                }}
              >
                Pause Training
              </a>
            );
          } else {
            if (record.stage === "finished") {
              return <></>;
            } else {
              return (
                <a
                  onClick={() => {
                    dispatch(setIsRunning(true));
                    dispatch(
                      addToQueue({
                        ID: record.ID,
                        type: "trainingRun",
                        name: record.configuration.name,
                      })
                    );
                  }}
                >
                  Start Training
                </a>
              );
            }
          }
        };

        return (
          <Space size="middle">
            <a
              onClick={() => {
                selectTrainingRun(record);
              }}
            >
              Select
            </a>
            {getRunningLink()}
            <Popconfirm
              title="Are you sure you want to delete this training run?"
              onConfirm={() => {
                removeTrainingRun(record);
                pollTrainingRuns();
              }}
              okText="Yes"
              cancelText="No"
              disabled={isRunning}
            >
              {isRunning ? (
                <Typography.Text disabled>Delete</Typography.Text>
              ) : (
                <a href="#">Delete</a>
              )}
            </Popconfirm>
          </Space>
        );
      },
    },
  ];

  useInterval(pollTrainingRuns, POLL_LOGFILE_INTERVALL);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>Training Runs</Breadcrumb.Item>
      </Breadcrumb>
      <Card
        title={`Your Models in Training`}
        bodyStyle={{ display: "flex", width: "100%" }}
        className="dataset-card-wrapper"
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button onClick={createRun} style={{ marginRight: 8 }}>
              Train New Model
            </Button>
          </div>
          <Table
            bordered
            style={{ width: "100%" }}
            columns={columns}
            pagination={defaultPageOptions}
            dataSource={trainingRuns.map(
              (trainingRun: TrainingRunInterface) => {
                return {
                  ...trainingRun,
                  key: trainingRun.ID,
                };
              }
            )}
          ></Table>
        </div>
      </Card>
    </>
  );
}
