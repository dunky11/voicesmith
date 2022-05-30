import React, { useEffect, useState, useRef, ReactElement } from "react";
import {
  Row,
  Col,
  Card,
  Form,
  Select,
  Input,
  Button,
  Table,
  Slider,
  Breadcrumb,
  Space,
  Tabs,
  Typography,
} from "antd";
import { FormInstance } from "rc-field-form";
import { Link } from "react-router-dom";
import LogPrinter from "../../components/log_printer/LogPrinter";
import AudioBottomBar from "../../components/audio_player/AudioBottomBar";
import { createUseStyles } from "react-jss";
import {
  AudioSynthInterface,
  ModelInterface,
  ModelSpeakerInterface,
  SynthConfigInterface,
} from "../../interfaces";
import { SERVER_URL, defaultPageOptions } from "../../config";
import {
  FETCH_AUDIOS_SYNTH_CHANNEL,
  GET_AUDIO_DATA_URL_CHANNEL,
  EXPORT_FILES_CHANNEL,
  REMOVE_AUDIOS_SYNTH_CHANNEL,
} from "../../channels";
import { MODELS_ROUTE } from "../../routes";

const { ipcRenderer } = window.require("electron");

const useStyles = createUseStyles({});

export default function Synthesize({
  selectedModel,
  ...props
}: {
  selectedModel: ModelInterface | null;
}): ReactElement {
  const classes = useStyles();
  const [selectedAudios, setSelectedAudios] = useState<AudioSynthInterface[]>(
    []
  );
  const playFuncRef = useRef<null | (() => void)>(null);
  const initialValues: SynthConfigInterface = {
    text: "",
    speakerID:
      selectedModel !== null ? selectedModel.speakers[0].speakerID : null,
    talkingSpeed: 1.0,
  };
  const isMounted = useRef(false);
  const [isLoading, setIsLoading] = useState(true);
  const [audios, setAudios] = useState<AudioSynthInterface[]>([]);
  const [audioDataURL, setAudioDataURL] = useState<string | null>(null);
  const formRef = useRef<FormInstance | null>();
  const [formValues, setFormValues] =
    useState<SynthConfigInterface>(initialValues);

  const fetchAudios = (playOnLoad: boolean) => {
    ipcRenderer
      .invoke(FETCH_AUDIOS_SYNTH_CHANNEL.IN)
      .then((audios: AudioSynthInterface[]) => {
        if (!isMounted.current) {
          return;
        }
        setAudios(audios);
        if (playOnLoad) {
          loadAudio(audios[0].filePath);
        }
      });
  };

  const openModel = () => {
    if (selectedModel === null || formValues.speakerID === null) {
      return;
    }
    setIsLoading(true);
    const ajax = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("modelID", String(selectedModel.ID));
    ajax.open("POST", `${SERVER_URL}/open-model`);
    ajax.onload = () => {
      setIsLoading(false);
    };
    ajax.send(formData);
  };

  const closeModel = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", `${SERVER_URL}/close-model`);
    ajax.send();
  };

  const onSynthesize = () => {
    if (selectedModel === null || formValues.speakerID === null) {
      return;
    }
    setIsLoading(true);
    const ajax = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("modelID", String(selectedModel.ID));
    formData.append("talkingSpeed", String(formValues.talkingSpeed));
    formData.append("text", formValues.text);
    formData.append("speakerID", String(formValues.speakerID));
    ajax.open("POST", `${SERVER_URL}/synthesize`);
    ajax.onload = () => {
      fetchAudios(true);
      setIsLoading(false);
    };
    ajax.onerror = () => {
      setIsLoading(false);
    };
    ajax.send(formData);
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

  const onExportSelected = () => {
    ipcRenderer.invoke(
      EXPORT_FILES_CHANNEL.IN,
      selectedAudios.map((audio: AudioSynthInterface) => audio.filePath)
    );
  };

  const onRemoveSelected = () => {
    ipcRenderer
      .invoke(REMOVE_AUDIOS_SYNTH_CHANNEL.IN, selectedAudios)
      .then(() => {
        fetchAudios(false);
      });
  };

  const columns = [
    {
      title: "Text",
      key: "text",
      render: (text: any, record: AudioSynthInterface) => (
        <Typography.Text>{record.text}</Typography.Text>
      ),
    },
    {
      title: "Model",
      dataIndex: "modelName",
      key: "model",
    },
    {
      title: "Speaker",
      dataIndex: "speakerName",
      key: "speakerName",
    },
    {
      title: "Created At",
      dataIndex: "createdAt",
      key: "createdAt",
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: AudioSynthInterface) => (
        <Space size="middle">
          <a
            onClick={() => {
              loadAudio(record.filePath);
            }}
          >
            Play
          </a>
        </Space>
      ),
    },
  ];

  useEffect(() => {
    isMounted.current = true;
    fetchAudios(false);
    return () => {
      closeModel();
      isMounted.current = false;
    };
  }, []);

  useEffect(() => {
    console.log(selectedModel);
    if (selectedModel === null) {
      return;
    }
    openModel();
    setFormValues({
      ...formValues,
      speakerID: selectedModel.speakers[0].speakerID,
    });
  }, [selectedModel]);

  return (
    <>
      <Breadcrumb style={{ marginBottom: 8 }}>
        <Breadcrumb.Item>
          <Link to={MODELS_ROUTE.SELECTION.ROUTE}>Models</Link>
        </Breadcrumb.Item>
        <Breadcrumb.Item>Synthesize</Breadcrumb.Item>
      </Breadcrumb>
      <Row gutter={[16, 24]}>
        <Col span={8}>
          <Card
            bodyStyle={{ paddingTop: 8 }}
            actions={[
              <div
                key="submit-button"
                style={{ marginLeft: 24, marginRight: 24 }}
              >
                <Button
                  key="submit-button"
                  htmlType="submit"
                  disabled={
                    isLoading ||
                    formValues.text.trim().length === 0 ||
                    formValues.speakerID === null
                  }
                  loading={isLoading}
                  type="primary"
                  block
                  onClick={onSynthesize}
                >
                  Convert
                </Button>
              </div>,
            ]}
          >
            <Tabs defaultActiveKey="Overview">
              <Tabs.TabPane tab="Overview" key="overview">
                <Form
                  layout="vertical"
                  ref={(node) => {
                    formRef.current = node;
                  }}
                  onValuesChange={(_, values) => {
                    setFormValues(values);
                  }}
                  initialValues={initialValues}
                >
                  <Form.Item name="speakerID" label="Speaker">
                    <Select style={{ width: "100%", marginBottom: 8 }}>
                      {selectedModel &&
                        selectedModel.speakers.map(
                          (speaker: ModelSpeakerInterface) => (
                            <Select.Option
                              key={speaker.speakerID}
                              value={speaker.speakerID}
                            >
                              {speaker.name}
                            </Select.Option>
                          )
                        )}
                    </Select>
                  </Form.Item>
                  <Form.Item label="Text:" name="text">
                    <Input.TextArea name="text" rows={4} />
                  </Form.Item>
                  <Form.Item label="Talking Speed:" name="talkingSpeed">
                    <Slider min={0.2} max={3.0} step={0.01} />
                  </Form.Item>
                </Form>
              </Tabs.TabPane>
              <Tabs.TabPane tab="Log" key="log">
                {selectedModel && (
                  <LogPrinter
                    name={selectedModel.name}
                    logFileName="synthesize.txt"
                    type="model"
                  />
                )}
              </Tabs.TabPane>
            </Tabs>
          </Card>
        </Col>
        <Col span={16}>
          <Card>
            <div style={{ marginBottom: 16, display: "flex" }}>
              <Button
                style={{ marginRight: 8 }}
                disabled={selectedAudios.length === 0}
                onClick={onExportSelected}
              >
                Export Selected
              </Button>
              <Button
                onClick={onRemoveSelected}
                disabled={selectedAudios.length === 0}
              >
                Remove Selected
              </Button>
            </div>
            <Table
              size="small"
              pagination={defaultPageOptions}
              bordered
              style={{ width: "100%" }}
              columns={columns}
              dataSource={audios.map((audio: AudioSynthInterface) => {
                return {
                  ...audio,
                  key: audio.ID,
                };
              })}
              rowSelection={{
                selectedRowKeys: selectedAudios.map(
                  (audio: AudioSynthInterface) => audio.ID
                ),
                onChange: (
                  selectedRowKeys: any[],
                  selectedRows: AudioSynthInterface[]
                ) => {
                  setSelectedAudios(selectedRows);
                },
              }}
            ></Table>
          </Card>
        </Col>
      </Row>
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
