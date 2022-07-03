import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function RunValidationEveryInput({
  disabled,
  name,
  docsUrl,
}: {
  disabled: boolean;
  name: string;
  docsUrl: string | null;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          Run Validation Every
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name={name}
    >
      <InputNumber disabled={disabled} step={1} min={0} />
    </Form.Item>
  );
}

RunValidationEveryInput.defaultProps = {
  disabled: false,
  docsUrl: null,
};
