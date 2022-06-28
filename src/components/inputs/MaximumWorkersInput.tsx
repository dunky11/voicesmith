import React, { ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function MaximumWorkersInput({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          Maximum Worker Count
          <HelpIcon
            content={
              <Typography>
                How many CPU cores to use. Generally this should be lower than
                the number of cores your CPU has. If set to "AUTO", this will be
                set to maximum(1, number of CPU cores - 1).
              </Typography>
            }
            style={{ marginLeft: 8 }}
          />
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
};
