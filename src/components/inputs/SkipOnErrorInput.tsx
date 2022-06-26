import React, { ReactElement } from "react";
import { Form, Select } from "antd";

export default function SkipOnErrorInput({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  return (
    <Form.Item label="On Error Ignore Sample" name="skipOnError">
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
