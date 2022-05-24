import React, { useState, useEffect, useRef } from "react";
import {
  Card,
  Button,
  Table,
  Space,
  Typography,
  Progress,
  Breadcrumb,
} from "antd";
import { Link, useHistory } from "react-router-dom";
import Speaker from "./Speaker";
import InfoButton from "./InfoButton";
import { defaultPageOptions } from "../../config";
import { stringCompare, numberCompare } from "../../utils";
import { DatasetInterface, SpeakerInterface } from "../../interfaces";
const { ipcRenderer } = window.require("electron");

export default function Dataset({ datasetID }: { datasetID: number | null }) {
  const isMounted = useRef(false);
  const history = useHistory();
  const [isDisabled, setIsDisabled] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<number[]>([]);
  const [dirProgress, setDirProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);
  const [selectedSpeakerID, setSelectedSpeakerID] = useState<null | number>(
    null
  );
  const [dataset, setDataset] = useState<DatasetInterface | null>(null);
  const [hasLoaded, setHasLoaded] = useState<boolean>(false);

  let totalSampleCount = 0;
  dataset?.speakers.forEach((speaker) => {
    totalSampleCount += speaker.samples.length;
  });

  const onSpeakerNameEdit = (speakerID: number, newName: string) => {
    if (!speakerNameIsValid(newName)) {
      return;
    }

    ipcRenderer
      .invoke("edit-speaker-name", speakerID, newName)
      .then(fetchDataset);
  };

  const onRemoveSpeakers = () => {
    ipcRenderer
      .invoke("remove-speakers", datasetID, selectedRowKeys)
      .then(fetchDataset);
  };

  const addSpeaker = (speakerName: string) => {
    ipcRenderer
      .invoke("add-speaker", speakerName, datasetID)
      .then(fetchDataset);
  };

  const speakerNameIsValid = (speakerName: string) => {
    if (dataset === null) {
      return false;
    }
    for (const speaker of dataset.speakers) {
      if (speaker.name === speakerName) {
        return false;
      }
    }
    return true;
  };

  const onAddEmptySpeakerClick = () => {
    let index = 1;
    let speakerName = `Speaker ${index}`;
    while (!speakerNameIsValid(speakerName)) {
      index += 1;
      speakerName = `Speaker ${index}`;
    }
    addSpeaker(speakerName);
  };

  const onAddSpeakers = () => {
    if (datasetID === null) {
      return;
    }
    ipcRenderer.removeAllListeners("pick-speakers-progress-reply");
    ipcRenderer.removeAllListeners("pick-speakers-reply");
    ipcRenderer.on(
      "pick-speakers-progress-reply",
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
    ipcRenderer.once("pick-speakers-reply", () => {
      if (!isMounted.current) {
        return;
      }
      setIsDisabled(false);
      setDirProgress(null);
      fetchDataset();
    });
    setIsDisabled(true);
    ipcRenderer.send("pick-speakers", datasetID);
  };

  const columns = [
    {
      title:
        "Name" +
        (dataset === null || dataset?.speakers.length === 0
          ? ""
          : ` (${dataset.speakers.length} Speakers Total)`),
      key: "name",
      render: (text: any, record: SpeakerInterface) => (
        <Typography.Text
          editable={{
            tooltip: false,
            onChange: (newName: string) => {
              onSpeakerNameEdit(record.ID, newName);
            },
          }}
        >
          {record.name}
        </Typography.Text>
      ),
      sorter: {
        compare: (a: SpeakerInterface, b: SpeakerInterface) =>
          stringCompare(a.name, b.name),
      },
    },
    {
      title:
        "Number of Samples" +
        (totalSampleCount === 0 ? "" : ` (${totalSampleCount} Total)`),
      key: "samplecount",
      sorter: {
        compare: (a: SpeakerInterface, b: SpeakerInterface) => {
          return numberCompare(a.samples.length, b.samples.length);
        },
      },
      render: (text: any, record: SpeakerInterface) => (
        <Typography.Text>{record.samples.length}</Typography.Text>
      ),
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: any) => (
        <Space size="middle">
          <a
            onClick={() => {
              setSelectedSpeakerID(record.ID);
            }}
          >
            Select
          </a>
        </Space>
      ),
    },
  ];

  const fetchDataset = () => {
    if (datasetID === null) {
      return;
    }
    ipcRenderer
      .invoke("fetch-dataset", datasetID)
      .then((dataset: DatasetInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (!hasLoaded) {
          setHasLoaded(true);
        }
        setDataset(dataset);
      });
  };

  const getSelectedSpeaker = () => {
    if (dataset === null) {
      return null;
    }
    for (const speaker of dataset.speakers) {
      if (speaker.ID === selectedSpeakerID) {
        return speaker;
      }
    }
    return null;
  };

  const onBackClick = () => {
    history.push("/datasets/dataset-selection");
  };

  useEffect(() => {
    if (datasetID === null) {
      return;
    }
    fetchDataset();
  }, [datasetID]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      ipcRenderer.removeAllListeners("pick-speakers-progress-reply");
      ipcRenderer.removeAllListeners("pick-speakers-reply");
    };
  }, []);

  return (
    <>
      {selectedSpeakerID === null ? (
        <>
          <Breadcrumb style={{ marginBottom: 8 }}>
            <Breadcrumb.Item>
              <Link to="/datasets/dataset-selection">Datasets</Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>{dataset?.name}</Breadcrumb.Item>
          </Breadcrumb>
          <Card
            title="Add Speakers to your Model"
            actions={[
              <div
                key="next-button-wrapper"
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  marginRight: 24,
                }}
              >
                <Button onClick={onBackClick}>Back</Button>
              </div>,
            ]}
          >
            <div style={{ width: "100%" }}>
              <div style={{ display: "flex", marginBottom: 16 }}>
                <Button
                  onClick={onAddEmptySpeakerClick}
                  style={{ marginRight: 8 }}
                  disabled={
                    isDisabled || !hasLoaded || dataset?.referencedBy !== null
                  }
                >
                  Add Empty Speaker
                </Button>
                <Button
                  onClick={onAddSpeakers}
                  style={{ marginRight: 8 }}
                  disabled={
                    isDisabled || !hasLoaded || dataset?.referencedBy !== null
                  }
                  loading={dirProgress !== null}
                >
                  Add Speakers From Folders
                </Button>
                <Button
                  onClick={onRemoveSpeakers}
                  disabled={
                    selectedRowKeys.length === 0 ||
                    isDisabled ||
                    !hasLoaded ||
                    dataset?.referencedBy !== null
                  }
                  style={{ marginRight: 8 }}
                >
                  Remove Selected
                </Button>
                <InfoButton></InfoButton>
              </div>
              {dirProgress !== null && (
                <Progress
                  percent={(dirProgress.current / dirProgress.total) * 100}
                  style={{ borderRadius: 0 }}
                  showInfo={false}
                ></Progress>
              )}
              <Table
                size="small"
                pagination={defaultPageOptions}
                bordered
                style={{ width: "100%" }}
                columns={columns}
                dataSource={dataset?.speakers.map(
                  (speaker: SpeakerInterface) => ({
                    ...speaker,
                    key: speaker.ID,
                  })
                )}
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
      ) : (
        <Speaker
          datasetID={datasetID}
          datasetName={dataset !== null ? dataset.name : null}
          speaker={getSelectedSpeaker()}
          setSelectedSpeakerID={setSelectedSpeakerID}
          fetchDataset={fetchDataset}
          referencedBy={dataset?.referencedBy}
        ></Speaker>
      )}
    </>
  );
}
