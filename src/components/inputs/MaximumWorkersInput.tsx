import React from "react";
import { Form, Select } from "antd";

export default function MaximumWorkersInput({
  disabled,
}: {
  disabled: boolean;
}) {
  return (
    <Form.Item label="Maximum Number of Workers" name="maximumWorkers">
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
};
