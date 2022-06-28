import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function TrainingStepsInput({
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
          Training Steps
          <HelpIcon
            content={
              <Typography>
                Number of times the model will be updated. A higher value means
                more time spend on training, a potentially better model, but
                also more risk of overfitting.
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

TrainingStepsInput.defaultProps = {
  disabled: false,
};
