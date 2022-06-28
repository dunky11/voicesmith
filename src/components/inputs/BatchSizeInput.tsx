import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function BatchSizeInput({
  disabled,
  name,
}: {
  disabled: boolean;
  name: string;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          Batch Size
          <HelpIcon
            content={
              <Typography>
                Number of samples to be fed through the model at once. A smaller
                value means less RAM/VRAM usage but potentially more training
                steps needed for the model to converge.
              </Typography>
            }
            style={{ marginLeft: 8 }}
          />
        </Typography.Text>
      }
      name={name}
    >
      <InputNumber disabled={disabled} step={1} min={0} />
    </Form.Item>
  );
}

BatchSizeInput.defaultProps = {
  disabled: false,
};
