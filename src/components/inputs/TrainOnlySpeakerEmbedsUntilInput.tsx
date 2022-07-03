import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function TrainOnlySpeakerEmbedsUntilInput({
  disabled,
  name,
  rules,
  docsUrl,
}: {
  disabled: boolean;
  name: string;
  rules: any[];
  docsUrl: string | null;
}): ReactElement {
  return (
    <Form.Item
      rules={rules}
      label={
        <Typography.Text>
          Train Only Speaker Embeds Until
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name={name}
    >
      <InputNumber disabled={disabled} step={1} min={0} />
    </Form.Item>
  );
}

TrainOnlySpeakerEmbedsUntilInput.defaultProps = {
  disabled: false,
  rules: [],
  docsUrl: null,
};
