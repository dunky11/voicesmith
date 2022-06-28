import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function LearningRateInput({
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
          Learning Rate
          <HelpIcon
            content={
              <Typography>
                Learning rate to use during training. A higher rate means faster
                but more unstable training. Usually set to a very small value
                less than one but higher than zero.
              </Typography>
            }
            style={{ marginLeft: 8 }}
          />
        </Typography.Text>
      }
      name={name}
      rules={[
        () => ({
          validator(_, value) {
            if (value === 0) {
              return Promise.reject(
                new Error("Learning rate must be greater than zero")
              );
            }
            return Promise.resolve();
          },
        }),
      ]}
    >
      <InputNumber disabled={disabled} step={0.0001} min={0} />
    </Form.Item>
  );
}

LearningRateInput.defaultProps = {
  disabled: false,
};
