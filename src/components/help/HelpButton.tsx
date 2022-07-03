import React, { ReactElement } from "react";
import { Button } from "antd";
import { QuestionOutlined } from "@ant-design/icons";
import { documentationUrl } from "../../config";
const { shell } = window.require("electron");

export default function HelpButton({
  children,
  docsUrl,
  ...rest
}: {
  children: ReactElement | string;
  docsUrl: string;
  [x: string]: any;
}): ReactElement {
  return (
    <div style={{ display: "flex" }}>
      <Button
        {...rest}
        style={{
          borderTopRightRadius: 0,
          borderBottomRightRadius: 0,
        }}
        icon={<QuestionOutlined />}
        onClick={() => {
          shell.openExternal(`${documentationUrl}${docsUrl}`);
        }}
      >
        {children}
      </Button>
    </div>
  );
}
