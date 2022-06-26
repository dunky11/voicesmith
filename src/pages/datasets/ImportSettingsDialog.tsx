import React, { ReactElement } from "react";
import { Modal, Form } from "antd";
import LanguageSelect from "../../components/inputs/LanguageSelect";
import { useDispatch, useSelector } from "react-redux";
import { editImportSettings } from "../../features/importSettings";
import { RootState } from "../../app/store";

export default function ImportSettingsDialog({
  open,
  onClose,
  onOk,
}: {
  open: boolean;
  onClose: () => void;
  onOk: () => void;
}): ReactElement {
  const dispatch = useDispatch();
  const importSettings = useSelector(
    (state: RootState) => state.importSettings
  );

  return (
    <Modal
      title="Import Settings"
      visible={open}
      onOk={onOk}
      okText="Pick Speakers"
      onCancel={onClose}
    >
      <Form layout="vertical">
        <Form.Item label="Language">
          <LanguageSelect
            value={importSettings.language}
            onChange={(lang) => {
              dispatch(
                editImportSettings({ ...importSettings, language: lang })
              );
            }}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
}
