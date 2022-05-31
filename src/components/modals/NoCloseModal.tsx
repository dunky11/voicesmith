import React, { ReactElement } from "react";
import { Modal } from "antd";

export default function NoCloseModal({
  visible,
  title,
  children,
}: {
  visible: boolean;
  title: string;
  children: React.ReactNode;
}): ReactElement {
  return (
    <Modal
      title={title}
      visible={visible}
      onOk={null}
      onCancel={null}
      // TODO find way to remove cursor on close icon area hover
      closeIcon={<></>}
      footer={null}
    >
      {children}
    </Modal>
  );
}
