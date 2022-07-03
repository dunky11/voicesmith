import React, { ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function MaximumWorkersInput({
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
          Maximum Worker Count
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name="maximumWorkers"
    >
      <Select style={{ width: 200 }} disabled={disabled}>
        <Select.Option value={-1}>Auto</Select.Option>
        {Array.from(Array(64 + 1).keys())
          .slice(1)
          .map((el) => (
            <Select.Option key={el} value={el}>
              {el}
            </Select.Option>
          ))}
      </Select>
    </Form.Item>
  );
}

MaximumWorkersInput.defaultProps = {
  disabled: false,
  docsUrl: null,
};
