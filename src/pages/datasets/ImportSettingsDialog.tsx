import React from "react";
import { Modal } from "antd";
import LanguageSelect from "../../components/inputs/LanguageSelect";

export default function ImportSettingsDialog({
  open,
  onClose,
  onOk,
}: {
  open: boolean;
  onClose: () => void;
  onOk: () => void;
}) {
  return (
    <Modal
      title="Import Settings"
      visible={open}
      onOk={onOk}
      okText="Pick Speakers"
      onCancel={onClose}
    >
      <LanguageSelect value="en" onChange={() => {}} />
    </Modal>
  );
}
