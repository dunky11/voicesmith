import React, { ReactElement } from "react";
import { Modal } from "antd";
import { createUseStyles } from "react-jss";

const useStyles = createUseStyles({
  footerWrapper: {
    display: "flex",
    justifyContent: "flex-end",
  },
});

const getFooter = (buttons: React.ReactNode[]) => {
  const classes = useStyles();
  return (
    <div className={classes.footerWrapper}>
      {buttons.map((button, index) => (
        <div
          key={index}
          style={{ marginRight: index === buttons.length - 1 ? 0 : 8 }}
        >
          {button}
        </div>
      ))}
    </div>
  );
};

export default function NoCloseModal({
  visible,
  title,
  children,
  buttons,
}: {
  visible: boolean;
  title: string;
  children: React.ReactNode;
  buttons: React.ReactNode[] | null;
}): ReactElement {
  return (
    <Modal
      title={title}
      visible={visible}
      onOk={null}
      onCancel={null}
      // TODO find way to remove cursor on close icon area hover
      closeIcon={<></>}
      footer={buttons === null ? null : getFooter(buttons)}
    >
      {children}
    </Modal>
  );
}

NoCloseModal.defaultProps = {
  buttons: null,
};
