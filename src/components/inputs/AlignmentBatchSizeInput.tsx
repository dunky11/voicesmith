import React, { ReactElement } from "react";
import { Form, InputNumber, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function AlignmentBatchSizeInput({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  return (
    <Form.Item
      label={
        <Typography.Text>
          Forced Aligner Batch Size
          <HelpIcon
            content={
              <Typography>
                How many samples to process at once during aligning the text
                with the audio files. Higher numbers mean faster alignment but
                larger disk space requirements together with a higher chance for
                things to go wrong. Recommended to be set to a very high value
                like 200.000.
              </Typography>
            }
            style={{ marginLeft: 8 }}
          />
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
};
