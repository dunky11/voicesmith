import React, { useEffect, useState, useRef, ReactElement } from "react";
import {
  Table,
  Button,
  Card,
  Space,
  Breadcrumb,
  Popconfirm,
  Typography,
} from "antd";
import { useSelector, useDispatch } from "react-redux";
import HelpIcon from "../../components/help/HelpIcon";
import BreadcrumbItem from "../../components/breadcrumb/BreadcrumbItem";
import { RunInterface, TrainingRunInterface } from "../../interfaces";
import { POLL_LOGFILE_INTERVALL, defaultPageOptions } from "../../config";
import { useInterval, stringCompare, getStateTag } from "../../utils";
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
  const runManager = useSelector((state: RootState) => {
    return state.runManager;
  });
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
      render: (text: any, record: any) =>
        getStateTag(record, runManager.isRunning, runManager.queue),
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
        <BreadcrumbItem>Training Runs</BreadcrumbItem>
      </Breadcrumb>
      <Card
        title={
          <div>
            Your Models in Training
            <HelpIcon style={{ marginLeft: 8 }} docsUrl="/usage/training" />
          </div>
        }
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
