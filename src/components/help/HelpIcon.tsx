import React, { ReactElement } from "react";
import { Button } from "antd";
import { QuestionOutlined } from "@ant-design/icons";
import { documentationUrl } from "../../config";
const { shell } = window.require("electron");

export default function HelpIcon({
  docsUrl,
  ...props
}: {
  docsUrl: string;
  [x: string]: any;
}): ReactElement {
  return (
    <Button
      {...props}
      size="small"
      onClick={() => {
        shell.openExternal(`${documentationUrl}${docsUrl}`);
      }}
      shape="circle"
      icon={<QuestionOutlined />}
    />
  );
}
