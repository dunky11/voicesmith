import React, { useEffect, useRef, useState } from "react";
import {
  Table,
  Button,
  Card,
  Space,
  Breadcrumb,
  Popconfirm,
  Typography,
  Progress,
} from "antd";
import InfoButton from "./InfoButton";
import { DatasetInterface } from "../../interfaces";
import { numberCompare, stringCompare } from "../../utils";
const { ipcRenderer } = window.require("electron");

export default function DatasetSelection({
  setSelectedDatasetID,
}: {
  setSelectedDatasetID: (ID: number | null) => void;
}) {
  const isMounted = useRef(false);
  const [datasets, setDatasets] = useState<DatasetInterface[]>([]);
  const [isDisabled, setIsDisabled] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<number[]>([]);
  const [dirProgress, setDirProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);

  const getFirstPossibleName = () => {
    const names = datasets.map((dataset: DatasetInterface) => dataset.name);
    let i = 1;
    let name = `Dataset ${i}`;
    while (names.includes(name)) {
      i += 1;
      name = `Dataset ${i}`;
    }
    return name;
  };

  const datasetNameIsValid = (name: string) => {
    const names = datasets.map((dataset: DatasetInterface) => dataset.name);
    return !names.includes(name);
  };

  const onDatasetNameEdit = (datasetID: number, newName: string) => {
    if (!datasetNameIsValid(newName)) {
      return;
    }

    ipcRenderer
      .invoke("edit-dataset-name", datasetID, newName)
      .then(fetchDatasets);
  };

  const fetchDatasets = () => {
    ipcRenderer.invoke("fetch-datasets").then((ds: DatasetInterface[]) => {
      if (!isMounted.current) {
        return;
      }
      setDatasets(ds);
    });
  };

  const removeDataset = (ID: number) => {
    ipcRenderer.invoke("remove-dataset", ID).then(fetchDatasets);
  };

  const exportDatasets = () => {
    ipcRenderer.removeAllListeners("export-datasets-progress-reply");
    ipcRenderer.removeAllListeners("export-datasets-reply");
    const exportedDatasets = datasets.filter((dataset: DatasetInterface) =>
      selectedRowKeys.includes(dataset.ID)
    );
    ipcRenderer.on(
      "export-datasets-progress-reply",
      (event: any, current: number, total: number) => {
        if (!isMounted.current) {
          return;
        }
        setDirProgress({
          current,
          total,
        });
      }
    );
    ipcRenderer.once("export-datasets-reply", () => {
      if (!isMounted.current) {
        return;
      }
      setIsDisabled(false);
      setDirProgress(null);
    });
    setIsDisabled(true);
    ipcRenderer.send("export-datasets", exportedDatasets);
  };

  const createDataset = () => {
    const name = getFirstPossibleName();
    ipcRenderer.invoke("create-dataset", name).then(fetchDatasets);
  };

  const columns = [
    {
      title: "Name",
      key: "name",
      render: (text: any, record: DatasetInterface) => (
        <Typography.Text
          editable={{
            tooltip: false,
            onChange: (newName: string) => {
              onDatasetNameEdit(record.ID, newName);
            },
          }}
          disabled={isDisabled}
        >
          {record.name}
        </Typography.Text>
      ),
      sorter: {
        compare: (a: DatasetInterface, b: DatasetInterface) =>
          stringCompare(a.name, b.name),
      },
    },
    {
      title: "Number of Speakers",
      key: "speakerCount",
      dataIndex: "speakerCount",
      sorter: {
        compare: (a: DatasetInterface, b: DatasetInterface) => {
          if (a.speakerCount === undefined || b.speakerCount === undefined) {
            return 0;
          }
          return numberCompare(a.speakerCount, b.speakerCount);
        },
      },
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: DatasetInterface) => (
        // TODO find way to diable visually
        <Space size="middle">
          <a
            onClick={
              isDisabled
                ? undefined
                : () => {
                    setSelectedDatasetID(record.ID);
                  }
            }
          >
            Select
          </a>
          <Popconfirm
            title="Are you sure you want to delete this dataset?"
            onConfirm={() => {
              removeDataset(record.ID);
            }}
            okText="Yes"
            cancelText="No"
            disabled={isDisabled || record.referencedBy !== null}
          >
            {isDisabled || record.referencedBy !== null ? (
              <Typography.Text disabled>Delete</Typography.Text>
            ) : (
              <a href="#">Delete</a>
            )}
          </Popconfirm>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    isMounted.current = true;
    fetchDatasets();
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners("export-datasets-progress-reply");
      ipcRenderer.removeAllListeners("export-datasets-reply");
    };
  }, []);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>Datasets</Breadcrumb.Item>
      </Breadcrumb>

      <Card
        title={`Your Datasets`}
        bodyStyle={{ display: "flex", width: "100%" }}
      >
        <div style={{ width: "100%" }}>
          <div style={{ marginBottom: 16, display: "flex" }}>
            <Button
              onClick={exportDatasets}
              disabled={selectedRowKeys.length === 0 || isDisabled}
              style={{ marginRight: 8 }}
            >
              Export Selected Datasets
            </Button>
            <Button
              onClick={createDataset}
              style={{ marginRight: 8 }}
              disabled={isDisabled}
            >
              Create Empty Dataset
            </Button>
            <InfoButton />
          </div>
          {dirProgress !== null && (
            <Progress
              percent={(dirProgress.current / dirProgress.total) * 100}
              style={{ borderRadius: 0 }}
              showInfo={false}
            ></Progress>
          )}
          <Table
            bordered
            style={{ width: "100%" }}
            columns={columns}
            dataSource={datasets.map((ds: DatasetInterface) => {
              return {
                ...ds,
                key: ds.ID,
              };
            })}
            rowSelection={{
              selectedRowKeys,
              onChange: (selectedRowKeys: any[]) => {
                setSelectedRowKeys(selectedRowKeys);
              },
            }}
          ></Table>
        </div>
      </Card>
    </>
  );
}
