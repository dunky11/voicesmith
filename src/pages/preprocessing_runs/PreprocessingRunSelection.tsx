import React, { ReactElement, useEffect, useRef, useState } from "react";
import { SyncOutlined } from "@ant-design/icons";
import {
  Table,
  Button,
  Card,
  Space,
  Breadcrumb,
  Popconfirm,
  Typography,
  Tag,
} from "antd";
import { defaultPageOptions } from "../../config";
import { PreprocessingRunInterface, RunInterface } from "../../interfaces";
import { stringCompare } from "../../utils";
import {
  CREATE_PREPROCESSING_RUN_CHANNEL,
  EDIT_PREPROCESSING_RUN_NAME_CHANNEL,
  FETCH_PREPROCESSING_RUNS_CHANNEL,
  REMOVE_PREPROCESSING_RUN_CHANNEL,
} from "../../channels";
const { ipcRenderer } = window.require("electron");

const prettyType = (
  type: "textNormalizationRun" | "dSCleaningRun" | "sampleSplittingRun"
) => {
  switch (type) {
    case "textNormalizationRun":
      return "Text Normalization";
    case "dSCleaningRun":
      return "Dataset Cleaning";
    case "sampleSplittingRun":
      return "Sample Splitting";
    default:
      throw new Error(
        `No case selected in switch-statement, '${type}' is not a valid case ...`
      );
  }
};

export default function PreprocessingRunSelection({
  setSelectedPreprocessingRun,
  running,
  stopRun,
  continueRun,
}: {
  setSelectedPreprocessingRun: (
    preprocessingRun: PreprocessingRunInterface | null
  ) => void;
  running: RunInterface | null;
  stopRun: () => void;
  continueRun: (run: RunInterface) => void;
}): ReactElement {
  const isMounted = useRef(false);
  const [preprocessingRuns, setPreprocessingRuns] = useState<
    PreprocessingRunInterface[]
  >([]);
  const [isDisabled, setIsDisabled] = useState(false);

  const getFirstPossibleName = () => {
    const names = preprocessingRuns.map(
      (preprocessingRun: PreprocessingRunInterface) => preprocessingRun.name
    );
    let i = 1;
    let name = `Preprocessing Run ${i}`;
    while (names.includes(name)) {
      i += 1;
      name = `Preprocessing Run ${i}`;
    }
    return name;
  };

  const nameIsValid = (name: string) => {
    const names = preprocessingRuns.map(
      (preprocessingRun: PreprocessingRunInterface) => preprocessingRun.name
    );
    return !names.includes(name);
  };

  const onNameEdit = (
    preprocessingRun: PreprocessingRunInterface,
    newName: string
  ) => {
    if (!nameIsValid(newName)) {
      return;
    }
    ipcRenderer
      .invoke(EDIT_PREPROCESSING_RUN_NAME_CHANNEL.IN, preprocessingRun, newName)
      .then(fetchPreprocessingRuns);
  };

  const fetchPreprocessingRuns = () => {
    ipcRenderer
      .invoke(FETCH_PREPROCESSING_RUNS_CHANNEL.IN)
      .then((ds: PreprocessingRunInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setPreprocessingRuns(ds);
      });
  };

  const removePreprocessingRun = (
    preprocessingRun: PreprocessingRunInterface
  ) => {
    ipcRenderer
      .invoke(REMOVE_PREPROCESSING_RUN_CHANNEL.IN, preprocessingRun)
      .then(fetchPreprocessingRuns);
  };

  const createPreprocessingRun = (
    type: "textNormalizationRun" | "dSCleaningRun" | "sampleSplittingRun"
  ) => {
    const name = getFirstPossibleName();
    ipcRenderer
      .invoke(CREATE_PREPROCESSING_RUN_CHANNEL.IN, name, type)
      .then(fetchPreprocessingRuns);
  };

  const getIsRunning = (record: PreprocessingRunInterface) => {
    return (
      running !== null &&
      running.type === record.type &&
      record.ID === running.ID
    );
  };

  const columns = [
    {
      title: "Name",
      key: "name",
      render: (text: any, record: PreprocessingRunInterface) => (
        <Typography.Text
          editable={{
            tooltip: false,
            onChange: (newName: string) => {
              onNameEdit(record, newName);
            },
          }}
          disabled={isDisabled}
        >
          {record.name}
        </Typography.Text>
      ),
      sorter: {
        compare: (a: PreprocessingRunInterface, b: PreprocessingRunInterface) =>
          stringCompare(a.name, b.name),
      },
    },
    {
      title: "Type",
      key: "type",
      render: (text: any, record: PreprocessingRunInterface) => (
        <Typography.Text>{prettyType(record.type)}</Typography.Text>
      ),
      sorter: {
        compare: (a: PreprocessingRunInterface, b: PreprocessingRunInterface) =>
          stringCompare(prettyType(a.type), prettyType(b.type)),
      },
    },
    {
      title: "Stage",
      key: "stage",
      dataIndex: "stage",
      sorter: {
        compare: (a: PreprocessingRunInterface, b: PreprocessingRunInterface) =>
          stringCompare(a.stage, b.stage),
      },
    },
    {
      title: "State",
      key: "action",
      sorter: {
        compare: (a: PreprocessingRunInterface, b: PreprocessingRunInterface) =>
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
      render: (text: any, record: PreprocessingRunInterface) =>
        getIsRunning(record) ? (
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
        compare: (a: PreprocessingRunInterface, b: PreprocessingRunInterface) =>
          stringCompare(a.datasetName, b.datasetName),
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: PreprocessingRunInterface) => {
        const isRunning = getIsRunning(record);

        const getRunningLink = () => {
          if (isRunning) {
            return <a onClick={stopRun}>Pause Training</a>;
          } else {
            if (record.stage === "finished") {
              return <></>;
            } else {
              if (
                record.stage === "choose_samples" ||
                record.stage === "finished" ||
                record.stage == "not_started"
              ) {
                return <Typography.Text disabled>Continue Run</Typography.Text>;
              }
              return (
                <a
                  onClick={() => {
                    continueRun({ ID: record.ID, type: record.type });
                  }}
                >
                  Continue Run
                </a>
              );
            }
          }
        };

        return (
          <Space size="middle">
            <a
              onClick={
                isDisabled
                  ? undefined
                  : () => {
                      setSelectedPreprocessingRun(record);
                    }
              }
            >
              Select
            </a>
            {getRunningLink()}
            <Popconfirm
              title="Are you sure you want to delete this preprocessing run?"
              onConfirm={() => {
                removePreprocessingRun(record);
              }}
              okText="Yes"
              cancelText="No"
              disabled={isDisabled}
            >
              <a href="#">Delete</a>
            </Popconfirm>
          </Space>
        );
      },
    },
  ];

  useEffect(() => {
    isMounted.current = true;
    fetchPreprocessingRuns();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>Preprocessing Runs</Breadcrumb.Item>
      </Breadcrumb>
      <Card
        title={`Your Preprocessing Runs`}
        bodyStyle={{ display: "flex", width: "100%" }}
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              onClick={() => {
                createPreprocessingRun("textNormalizationRun");
              }}
              disabled={isDisabled}
              style={{ marginRight: 8 }}
            >
              New Text Normalization Run
            </Button>
            <Button
              onClick={() => {
                createPreprocessingRun("sampleSplittingRun");
              }}
              disabled={isDisabled}
            >
              New Sample Splitting Run
            </Button>
          </div>
          <Table
            bordered
            style={{ width: "100%" }}
            columns={columns}
            pagination={defaultPageOptions}
            dataSource={preprocessingRuns.map(
              (ds: PreprocessingRunInterface) => ({
                ...ds,
                key: `${ds.type}-${ds.ID}`,
              })
            )}
          ></Table>
        </div>
      </Card>
    </>
  );
}
