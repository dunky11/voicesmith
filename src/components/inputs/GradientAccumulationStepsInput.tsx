import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function GradientAccumulationStepsInput({
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
          Gradient Accumulation Steps
          <HelpIcon
            content={
              <Typography>
                Number of times samples are fed through the model per step. A
                higher values means potentially more stable training but will
                also increase the time spend on each step.
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

GradientAccumulationStepsInput.defaultProps = {
  disabled: false,
};
