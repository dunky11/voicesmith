import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function LearningRateInput({
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
          Learning Rate
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
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
  docsUrl: null,
};
