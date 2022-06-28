import React, { useState, useEffect, useRef, ReactElement } from "react";
import {
  Card,
  Button,
  Table,
  Space,
  Typography,
  Progress,
  Breadcrumb,
  InputRef,
} from "antd";
import { useHistory } from "react-router-dom";
import { IpcRendererEvent } from "electron";
import { createUseStyles } from "react-jss";
import BreadcrumbItem from "../../components/breadcrumb/BreadcrumbItem";
import { setNavIsDisabled } from "../../features/navigationSettingsSlice";
import Speaker from "./Speaker";
import InfoButton from "./InfoButton";
import ImportSettingsDialog from "./ImportSettingsDialog";
import { defaultPageOptions } from "../../config";
import {
  stringCompare,
  numberCompare,
  ISO6391_TO_NAME,
  getSearchableColumn,
} from "../../utils";
import { DatasetInterface, SpeakerInterface } from "../../interfaces";
import {
  ADD_SPEAKER_CHANNEL,
  REMOVE_SPEAKERS_CHANNEL,
  EDIT_SPEAKER_CHANNEL,
  PICK_SPEAKERS_CHANNEL,
  FETCH_DATASET_CHANNEL,
} from "../../channels";
import { DATASETS_ROUTE } from "../../routes";
import LanguageSelect from "../../components/inputs/LanguageSelect";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";

const { ipcRenderer } = window.require("electron");

const useStyles = createUseStyles({
  languageSelect: { width: 150 },
});

export default function Dataset({
  datasetID,
}: {
  datasetID: number | null;
}): ReactElement {
  const dispatch = useDispatch();
  const classes = useStyles();
  const isMounted = useRef(false);
  const history = useHistory();
  const importSettings = useSelector((root: RootState) => root.importSettings);
  const [importSettingsDialogIsOpen, setImportSettingsDialogIsOpen] =
    useState(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
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
  const [hasInitLoaded, setHasInitLoaded] = useState(false);
  const searchInput = useRef<InputRef>(null);

  let totalSampleCount = 0;
  dataset?.speakers.forEach((speaker) => {
    totalSampleCount += speaker.samples.length;
  });

  const onSpeakerNameEdit = (speaker: SpeakerInterface, newName: string) => {
    if (!speakerNameIsValid(newName)) {
      return;
    }
    onSpeakerChange({
      ...speaker,
      name: newName,
    });
  };

  const onSpeakerChange = (speaker: SpeakerInterface) => {
    ipcRenderer.invoke(EDIT_SPEAKER_CHANNEL.IN, speaker).then(fetchDataset);
  };

  const onRemoveSpeakers = () => {
    ipcRenderer
      .invoke(REMOVE_SPEAKERS_CHANNEL.IN, datasetID, selectedRowKeys)
      .then(fetchDataset);
  };

  const addSpeaker = (speakerName: string) => {
    ipcRenderer
      .invoke(ADD_SPEAKER_CHANNEL.IN, speakerName, datasetID)
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
    ipcRenderer.removeAllListeners(PICK_SPEAKERS_CHANNEL.REPLY);
    ipcRenderer.removeAllListeners(PICK_SPEAKERS_CHANNEL.PROGRESS_REPLY);
    ipcRenderer.on(
      PICK_SPEAKERS_CHANNEL.PROGRESS_REPLY,
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
    ipcRenderer.once(PICK_SPEAKERS_CHANNEL.REPLY, () => {
      if (!isMounted.current) {
        return;
      }
      dispatch(setNavIsDisabled(false));
      setIsLoading(false);
      setDirProgress(null);
      fetchDataset();
    });
    dispatch(setNavIsDisabled(true));
    setIsLoading(true);
    ipcRenderer.send(PICK_SPEAKERS_CHANNEL.IN, datasetID, importSettings);
  };

  const columns = [
    getSearchableColumn(
      {
        title:
          "Name" +
          (dataset === null || dataset?.speakers.length === 0
            ? ""
            : ` (${dataset.speakers.length} Speakers Total)`),
        key: "name",
        render: (text: any, record: SpeakerInterface) => (
          <Typography.Text
            editable={
              isLoading || isDisabled
                ? null
                : {
                    tooltip: false,
                    onChange: (newName: string) => {
                      onSpeakerNameEdit(record, newName);
                    },
                  }
            }
          >
            {record.name}
          </Typography.Text>
        ),
        sorter: {
          compare: (a: SpeakerInterface, b: SpeakerInterface) =>
            stringCompare(a.name, b.name),
        },
      },
      "name",
      searchInput
    ),
    getSearchableColumn(
      {
        title: `Language`,
        key: "language",
        sorter: {
          compare: (a: SpeakerInterface, b: SpeakerInterface) => {
            return stringCompare(a.language, b.language);
          },
        },
        render: (text: any, record: SpeakerInterface) => (
          <LanguageSelect
            className={classes.languageSelect}
            value={record.language}
            onChange={(lang: SpeakerInterface["language"]) => {
              onSpeakerChange({
                ...record,
                language: lang,
              });
            }}
            disabled={isDisabled || isLoading}
          />
        ),
      },
      "languageLong",
      searchInput
    ),
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
          {isLoading ? (
            <Typography.Text disabled>Select</Typography.Text>
          ) : (
            <a
              onClick={() => {
                setSelectedSpeakerID(record.ID);
              }}
            >
              Select
            </a>
          )}
        </Space>
      ),
    },
  ];

  const fetchDataset = () => {
    if (datasetID === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_DATASET_CHANNEL.IN, datasetID)
      .then((dataset: DatasetInterface) => {
        if (!isMounted.current) {
          return;
        }
        if (!hasInitLoaded) {
          setHasInitLoaded(true);
        }
        setDataset(dataset);
        setIsDisabled(dataset.referencedBy !== null);
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
    history.push(DATASETS_ROUTE.SELECTION.ROUTE);
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
      ipcRenderer.removeAllListeners(PICK_SPEAKERS_CHANNEL.REPLY);
      ipcRenderer.removeAllListeners(PICK_SPEAKERS_CHANNEL.PROGRESS_REPLY);
    };
  }, []);

  return selectedSpeakerID === null ? (
    <>
      <ImportSettingsDialog
        open={importSettingsDialogIsOpen}
        onOk={() => {
          setImportSettingsDialogIsOpen(false);
          onAddSpeakers();
        }}
        onClose={() => {
          setImportSettingsDialogIsOpen(false);
        }}
      ></ImportSettingsDialog>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <BreadcrumbItem to={DATASETS_ROUTE.SELECTION.ROUTE}>
          Datasets
        </BreadcrumbItem>
        <BreadcrumbItem>{dataset?.name}</BreadcrumbItem>
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
            <Button onClick={onBackClick} disabled={isLoading}>
              Back
            </Button>
          </div>,
        ]}
      >
        <div style={{ width: "100%" }}>
          <div style={{ display: "flex", marginBottom: 16 }}>
            <Button
              onClick={onAddEmptySpeakerClick}
              style={{ marginRight: 8 }}
              disabled={isDisabled || isLoading || !hasInitLoaded}
            >
              Add Empty Speaker
            </Button>
            <Button
              onClick={() => {
                setImportSettingsDialogIsOpen(true);
              }}
              style={{ marginRight: 8 }}
              disabled={isDisabled || isLoading || !hasInitLoaded}
              loading={dirProgress !== null}
            >
              Add Speakers From Folders
            </Button>
            <Button
              onClick={onRemoveSpeakers}
              disabled={
                isDisabled ||
                selectedRowKeys.length === 0 ||
                isLoading ||
                !hasInitLoaded
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
            dataSource={dataset?.speakers.map((speaker: SpeakerInterface) => ({
              ...speaker,
              languageLong: ISO6391_TO_NAME[speaker.language],
              key: speaker.ID,
            }))}
            rowSelection={
              isLoading
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
  ) : (
    <Speaker
      datasetID={datasetID}
      datasetName={dataset !== null ? dataset.name : null}
      speaker={getSelectedSpeaker()}
      setSelectedSpeakerID={setSelectedSpeakerID}
      fetchDataset={fetchDataset}
      isDisabled={isDisabled}
    ></Speaker>
  );
}
