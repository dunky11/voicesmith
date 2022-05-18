import React, { useEffect, useRef, useState } from "react";
import {
  Breadcrumb,
  Card,
  Form,
  Input,
  FormInstance,
  Button,
  Row,
  Col,
} from "antd";
import RunCard from "../../components/cards/RunCard";
import { RunInterface, SettingsInterface } from "../../interfaces";
import { notifySave } from "../../utils";
import { createUseStyles } from "react-jss";

const { ipcRenderer } = window.require("electron");

const useStyles = createUseStyles({
  breadcrumb: { marginBottom: 8 },
});

export default function Settings({
  running,
  setNavIsDisabled,
}: {
  running: RunInterface | null;
  setNavIsDisabled: (navIsDisabled: boolean) => void;
}) {
  const classes = useStyles();
  const formRef = useRef<FormInstance | null>();
  const isMounted = useRef(false);
  const [isLoading, setIsLoading] = useState(false);

  const onFinish = () => {
    setIsLoading(true);
    setNavIsDisabled(true);
    ipcRenderer.removeAllListeners("save-settings-reply");
    ipcRenderer.on(
      "save-settings-reply",
      (
        _: any,
        message: {
          type: string;
        }
      ) => {
        switch (message.type) {
          case "finished": {
            setIsLoading(false);
            notifySave();
            setNavIsDisabled(false);
            break;
          }
          default: {
            throw new Error(
              `No case selected in switch-statement, ${message.type} is not a valid case ...`
            );
          }
        }
      }
    );
    ipcRenderer.send("save-settings", formRef.current.getFieldsValue());
  };

  const fetchConfig = () => {
    ipcRenderer.invoke("fetch-settings").then((settings: SettingsInterface) => {
      console.log(settings);
      formRef.current.setFieldsValue(settings);
    });
  };

  const onPickStorageClick = () => {
    ipcRenderer.invoke("pick-single-folder").then((dataPath: string | null) => {
      if (dataPath == null) {
        return;
      }
      formRef.current.setFieldsValue({ dataPath });
    });
  };

  const onSaveClick = () => {
    formRef.current.submit();
  };

  useEffect(() => {
    isMounted.current = true;
    fetchConfig();
    return () => {
      ipcRenderer.removeAllListeners("save-settings-reply");
      isMounted.current = false;
    };
  });

  return (
    <>
      <Breadcrumb className={classes.breadcrumb}>
        <Breadcrumb.Item>Settings</Breadcrumb.Item>
      </Breadcrumb>
      <Row>
        <Col span={12}>
          <RunCard
            disableFullHeight
            buttons={[
              <Button type="primary" disabled={isLoading} onClick={onSaveClick}>
                Save
              </Button>,
            ]}
            title="Settings"
          >
            <Form
              layout="vertical"
              ref={(node) => {
                formRef.current = node;
              }}
              onFinish={onFinish}
            >
              <Form.Item label="Storage Path" name="dataPath">
                <Input.Search
                  disabled={isLoading || running !== null}
                  readOnly
                  onSearch={onPickStorageClick}
                  enterButton="Pick Path"
                  loading={isLoading}
                ></Input.Search>
              </Form.Item>
            </Form>
          </RunCard>
        </Col>
      </Row>
    </>
  );
}
