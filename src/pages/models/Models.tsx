import React, { ReactElement, useEffect, useRef, useState } from "react";
import {
  Card,
  Popconfirm,
  Tag,
  Breadcrumb,
  Space,
  Table,
  Alert,
  Typography,
  Modal,
} from "antd";
import BreadcrumbItem from "../../components/breadcrumb/BreadcrumbItem";
import { defaultPageOptions } from "../../config";
import { ModelInterface, ModelSpeakerInterface } from "../../interfaces";
import { stringCompare } from "../../utils";
import { FETCH_MODELS_CHANNEL, REMOVE_MODEL_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");
const MAX_SHOW_SPEAKERS = 12;

export default function Models({
  onModelSelect,
  pushRoute,
}: {
  onModelSelect: (model: ModelInterface) => void;
  pushRoute: (route: string) => void;
}): ReactElement {
  const isMounted = useRef(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [deleteModelConfirmOpen, setDeleteModelConfirmOpen] = useState(false);
  const [models, setModels] = useState<ModelInterface[]>([]);
  const [modelToDeleteName, setModelToDeleteName] = useState<string>("");
  const modelToDeleteID = useRef<number | null>(null);

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      sorter: {
        compare: (a: ModelInterface, b: ModelInterface) =>
          stringCompare(a.name, b.name),
      },
    },
    {
      title: "Type",
      dataIndex: "type",
      key: "type",
      sorter: {
        compare: (a: ModelInterface, b: ModelInterface) =>
          stringCompare(a.type, b.type),
      },
    },
    {
      title: "Speakers",
      key: "action",
      render: (text: any, record: ModelInterface) => (
        <div
          style={{
            display: "flex",
            margin: -4,
            flexWrap: "wrap",
            maxWidth: 600,
          }}
        >
          {record.speakers
            .slice(0, MAX_SHOW_SPEAKERS)
            .map((speaker: ModelSpeakerInterface) => {
              return (
                <div style={{ margin: 4 }} key={speaker.name}>
                  <Tag style={{ margin: 0 }} color="blue">
                    {speaker.name}
                  </Tag>
                </div>
              );
            })}
          {Object.keys(record.speakers).length > 20 && (
            <div style={{ margin: 4 }}>
              <Tag style={{ margin: 0 }}>
                And {Object.keys(record.speakers).length - MAX_SHOW_SPEAKERS}{" "}
                more ...
              </Tag>
            </div>
          )}
        </div>
      ),
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: any) => (
        <Space size="middle">
          <a
            onClick={() => {
              onModelSelect(record);
            }}
          >
            Select
          </a>
          <Popconfirm
            title="Are you sure you want to delete this model?"
            onConfirm={() => {
              onModelDelete(record.ID, record.name);
            }}
            okText="Yes"
            cancelText="No"
          >
            <a href="#">Delete</a>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const onModelDelete = (ID: number, name: string) => {
    modelToDeleteID.current = ID;
    setModelToDeleteName(name);
    setDeleteModelConfirmOpen(true);
  };

  const handleDeleteModel = () => {
    if (modelToDeleteID.current === null) {
      return;
    }
    setDeleteModelConfirmOpen(false);
    ipcRenderer
      .invoke(REMOVE_MODEL_CHANNEL.IN, modelToDeleteID.current)
      .then(fetchModels);
  };

  const handleDeleteModelClose = () => {
    setDeleteModelConfirmOpen(false);
  };

  const fetchModels = () => {
    ipcRenderer
      .invoke(FETCH_MODELS_CHANNEL.IN)
      .then((models: ModelInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setModels(models);
        setHasLoaded(true);
      });
  };

  useEffect(() => {
    isMounted.current = true;
    fetchModels();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <BreadcrumbItem>Models</BreadcrumbItem>
      </Breadcrumb>
      <Modal
        title="Confirm Deletion"
        visible={deleteModelConfirmOpen}
        onOk={handleDeleteModel}
        onCancel={handleDeleteModelClose}
      >
        <p>
          Do you really want to delete the model &apos;
          {modelToDeleteName}&apos;?
        </p>
      </Modal>

      <Card title={`Your Trained Models`} bodyStyle={{ width: "100%" }}>
        {hasLoaded && models.length === 0 && (
          <Alert
            type="info"
            message={
              <Typography>
                No model trained yet, please click on{" "}
                <a
                  onClick={() => {
                    pushRoute("/datasets/dataset-selection");
                  }}
                >
                  Datasets
                </a>{" "}
                and add a dataset and then click on{" "}
                <a
                  onClick={() => {
                    pushRoute("/training-runs/run-selection");
                  }}
                >
                  Training Runs
                </a>{" "}
                in order to create your first model.
              </Typography>
            }
            style={{ marginBottom: 16 }}
          ></Alert>
        )}
        <Table
          bordered
          style={{ width: "100%" }}
          columns={columns}
          pagination={defaultPageOptions}
          dataSource={models.map((model: ModelInterface) => {
            return {
              key: model.ID,
              ...model,
            };
          })}
        ></Table>
      </Card>
    </>
  );
}
