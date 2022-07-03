import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function AlignmentBatchSizeInput({
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
          Forced Aligner Batch Size
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name="forcedAlignmentBatchSize"
    >
      <InputNumber
        precision={0}
        disabled={disabled}
        step={100}
        min={0}
      ></InputNumber>
    </Form.Item>
  );
}

AlignmentBatchSizeInput.defaultProps = {
  disabled: false,
  docsUrl: null,
};
