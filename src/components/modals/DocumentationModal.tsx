import React, { ReactElement } from "react";
import { Modal, Button } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { createUseStyles } from "react-jss";
import { RootState } from "../../app/store";
import { setIsOpen } from "../../features/documentationManagerSlice";

const useStyles = createUseStyles({
  wrapper: {
    height: "calc(100vh - 20px)",
    top: "20px",
    "& .ant-modal-content": {
      display: "flex",
      flexDirection: "column",
    },
  },
});

export default function DocumentationModal({
  children,
}: {
  children: ReactElement;
}): ReactElement {
  const classes = useStyles();
  const dispatch = useDispatch();
  const isOpen = useSelector(
    (root: RootState) => root.documentationManager.isOpen
  );
  const onClose = () => {
    dispatch(setIsOpen(false));
  };
  return (
    <Modal
      visible={isOpen}
      closable={false}
      width={"100vw"}
      onCancel={onClose}
      bodyStyle={{ overflowY: "scroll", margin: 0, padding: 0 }}
      className={classes.wrapper}
      footer={<Button onClick={onClose}>Cancel</Button>}
    >
      {children}
    </Modal>
  );
}
