import React, { useEffect, useState, useRef } from "react";
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
import { RunInterface, TrainingRunBasicInterface } from "../../interfaces";
import { POLL_LOGFILE_INTERVALL } from "../../config";
import { useInterval, stringCompare } from "../../utils";
const { ipcRenderer } = window.require("electron");

export default function RunSelection({
  removeTrainingRun,
  selectTrainingRun,
  running,
  stopRun,
  continueRun,
}: {
  removeTrainingRun: (ID: number) => void;
  selectTrainingRun: (ID: number) => void;
  running: RunInterface | null;
  stopRun: () => void;
  continueRun: (run: RunInterface) => void;
}) {
  const isMounted = useRef(false);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRunBasicInterface[]>(
    []
  );

  const getFirstPossibleName = () => {
    const names = trainingRuns.map(
      (preprocessingRun: TrainingRunBasicInterface) => preprocessingRun.name
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
    ipcRenderer
      .invoke("fetch-training-runs")
      .then((trainingRuns: TrainingRunBasicInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setTrainingRuns(trainingRuns);
      });
  };

  const createRun = () => {
    const name = getFirstPossibleName();
    ipcRenderer.invoke("create-training-run", name).then(pollTrainingRuns);
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      sorter: {
        compare: (a: TrainingRunBasicInterface, b: TrainingRunBasicInterface) =>
          stringCompare(a.name, b.name),
      },
    },
    {
      title: "Stage",
      dataIndex: "stage",
      key: "stage",
      sorter: {
        compare: (a: TrainingRunBasicInterface, b: TrainingRunBasicInterface) =>
          stringCompare(a.stage, b.stage),
      },
    },

    {
      title: "State",
      key: "action",
      sorter: {
        compare: (a: TrainingRunBasicInterface, b: TrainingRunBasicInterface) =>
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
      render: (text: any, record: TrainingRunBasicInterface) =>
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
      dataIndex: "datasetName",
      key: "datasetName",
      sorter: {
        compare: (a: TrainingRunBasicInterface, b: TrainingRunBasicInterface) =>
          stringCompare(a.datasetName, b.datasetName),
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: TrainingRunBasicInterface) => {
        const isRunning =
          running !== null &&
          running.type === "trainingRun" &&
          record.ID === running.ID;

        const getRunningLink = () => {
          if (isRunning) {
            return <a onClick={stopRun}>Pause Training</a>;
          } else {
            if (record.stage === "finished") {
              return <></>;
            } else {
              return (
                <a
                  onClick={() => {
                    continueRun({ ID: record.ID, type: "trainingRun" });
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
                selectTrainingRun(record.ID);
              }}
            >
              Select
            </a>
            {getRunningLink()}
            <Popconfirm
              title="Are you sure you want to delete this training run?"
              onConfirm={() => {
                removeTrainingRun(record.ID);
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
            dataSource={trainingRuns.map(
              (trainingRun: TrainingRunBasicInterface) => {
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
