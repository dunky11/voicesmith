import React, { ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function SkipOnErrorInput({
  disabled,
  docsUrl,
}: {
  disabled: boolean;
  docsUrl: string | null;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          On Error Ignore Sample
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name="skipOnError"
    >
      <Select style={{ width: 200 }} disabled={disabled}>
        <Select.Option value={true}>Yes</Select.Option>
        <Select.Option value={false}>No</Select.Option>
      </Select>
    </Form.Item>
  );
}

SkipOnErrorInput.defaultProps = {
  disabled: false,
  docsUrl: null,
};
