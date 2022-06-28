import React, { ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function SkipOnErrorInput({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          On Error Ignore Sample
          <HelpIcon
            content={
              <Typography>
                If set to "Yes" and a sample could not be loaded, it will be
                skipped. If set to "NO" and a sample could not be loaded, the
                run will be stopped and a message will be written into the log
                indicating which sample caused the issue.
              </Typography>
            }
            style={{ marginLeft: 8 }}
          />
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
};
