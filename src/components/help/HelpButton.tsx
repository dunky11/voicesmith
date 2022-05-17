import React, { useState } from "react";
import { Modal, Button } from "antd";
import { QuestionCircleOutlined } from "@ant-design/icons";
import { createUseStyles } from "react-jss";

const useStyles = createUseStyles({
  wrapper: {
    height: "calc(100vh - 20px)",
    top: "20px",
    "& .ant-modal-content": {
      height: "100%",
      display: "flex",
      flexDirection: "column",
    },
  },
});

export default function HelpButton({
  buttonText,
  children,
  style,
}: {
  modalTitle: string;
  buttonText: string;
  children: React.ReactNode;
  style: { [Key: string]: string | number };
}) {
  const classes = useStyles();
  const [visible, setVisible] = useState(false);
  return (
    <>
      <Button
        style={style}
        onClick={() => setVisible(true)}
        icon={<QuestionCircleOutlined />}
      >
        {buttonText}
      </Button>
      <Modal
        visible={visible}
        onOk={() => setVisible(false)}
        onCancel={() => setVisible(false)}
        width={1000}
        bodyStyle={{ overflowY: "scroll", paddingBottom: 24 }}
        className={classes.wrapper}
        footer={[
          <Button
            key="Ok"
            type="primary"
            onClick={() => {
              setVisible(false);
            }}
          >
            Ok
          </Button>,
        ]}
      >
        {children}
      </Modal>
    </>
  );
}

HelpButton.defaultProps = {
  buttonText: "Help",
  style: {},
  children: [],
};
