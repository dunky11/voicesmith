import React, { ReactElement, useEffect, useRef, useState } from "react";
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
import { IpcRendererEvent } from "electron";
import { useDispatch } from "react-redux";
import BreadcrumbItem from "../../components/breadcrumb/BreadcrumbItem";
import HelpIcon from "../../components/help/HelpIcon";
import {
  EDIT_DATASET_NAME_CHANNEL,
  FETCH_DATASETS_CHANNEL,
  FETCH_DATASETS_CHANNEL_TYPES,
  REMOVE_DATASET_CHANNEL,
  EXPORT_DATASET_CHANNEL,
  CREATE_DATASET_CHANNEL,
} from "../../channels";
import { defaultPageOptions } from "../../config";
import { DatasetInterface } from "../../interfaces";
import { numberCompare, stringCompare } from "../../utils";
import { setNavIsDisabled } from "../../features/navigationSettingsSlice";
const { ipcRenderer } = window.require("electron");

export default function DatasetSelection({
  setSelectedDatasetID,
}: {
  setSelectedDatasetID: (ID: number | null) => void;
}): ReactElement {
  const dispatch = useDispatch();
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
      .invoke(EDIT_DATASET_NAME_CHANNEL.IN, datasetID, newName)
      .then(fetchDatasets);
  };

  const fetchDatasets = () => {
    const args: FETCH_DATASETS_CHANNEL_TYPES["IN"]["ARGS"] = {
      ID: null,
      withSamples: false
    }
    ipcRenderer
      .invoke(FETCH_DATASETS_CHANNEL.IN, args)
      .then((ds: FETCH_DATASETS_CHANNEL_TYPES["IN"]["OUT"]) => {
        if (!isMounted.current) {
          return;
        }
        setDatasets(ds);
      });
  };

  const removeDataset = (ID: number) => {
    ipcRenderer.invoke(REMOVE_DATASET_CHANNEL.IN, ID).then(fetchDatasets);
  };

  const exportDatasets = () => {
    ipcRenderer.removeAllListeners(EXPORT_DATASET_CHANNEL.REPLY);
    ipcRenderer.removeAllListeners(EXPORT_DATASET_CHANNEL.PROGRESS_REPLY);
    const exportedDatasets = datasets.filter((dataset: DatasetInterface) =>
      selectedRowKeys.includes(dataset.ID)
    );
    ipcRenderer.on(
      EXPORT_DATASET_CHANNEL.PROGRESS_REPLY,
      (event: IpcRendererEvent, current: number, total: number) => {
        if (!isMounted.current) {
          return;
        }
        setDirProgress({
          current,
          total,
        });
      }
    );
    ipcRenderer.once(EXPORT_DATASET_CHANNEL.REPLY, () => {
      if (!isMounted.current) {
        return;
      }
      setIsDisabled(false);
      dispatch(setNavIsDisabled(false));
      setDirProgress(null);
    });
    setIsDisabled(true);
    dispatch(setNavIsDisabled(true));
    ipcRenderer.send(EXPORT_DATASET_CHANNEL.IN, exportedDatasets);
  };

  const createDataset = () => {
    const name = getFirstPossibleName();
    ipcRenderer.invoke(CREATE_DATASET_CHANNEL.IN, name).then(fetchDatasets);
  };

  const columns = [
    {
      title: "Name",
      key: "name",
      render: (text: any, record: DatasetInterface) => (
        <Typography.Text
          editable={
            isDisabled
              ? null
              : {
                tooltip: false,
                onChange: (newName: string) => {
                  onDatasetNameEdit(record.ID, newName);
                },
              }
          }
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
      sorter: {
        compare: (a: DatasetInterface, b: DatasetInterface) => {
          return numberCompare(a.speakers.length, b.speakers.length);
        },
      },
      render: (text: any, record: DatasetInterface) => (
        record.speakers.length
      ),
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: DatasetInterface) => (
        <Space size="middle">
          {isDisabled ? (
            <Typography.Text disabled>Select</Typography.Text>
          ) : (
            <a
              onClick={() => {
                setSelectedDatasetID(record.ID);
              }}
            >
              Select
            </a>
          )}
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
      ipcRenderer.removeAllListeners(EXPORT_DATASET_CHANNEL.REPLY);
      ipcRenderer.removeAllListeners(EXPORT_DATASET_CHANNEL.PROGRESS_REPLY);
    };
  }, []);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <BreadcrumbItem>Datasets</BreadcrumbItem>
      </Breadcrumb>
      <Card
        title={
          <div>
            Your Datasets
            <HelpIcon
              style={{ marginLeft: 8 }}
              docsUrl="/usage/dataset-importing"
            />
          </div>
        }
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
            pagination={defaultPageOptions}
            dataSource={datasets.map((ds: DatasetInterface) => {
              return {
                ...ds,
                key: ds.ID,
              };
            })}
            rowSelection={
              isDisabled
                ? null
                : {
                  selectedRowKeys,
                  onChange: (selectedRowKeys: any[]) => {
                    setSelectedRowKeys(selectedRowKeys);
                  },
                }
            }
          ></Table>
        </div>
      </Card>
    </>
  );
}
