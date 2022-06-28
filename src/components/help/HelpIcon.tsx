import React, { ReactElement } from "react";
import { QuestionCircleOutlined } from "@ant-design/icons";
import { Popover } from "antd";

export default function HelpIcon({
  title,
  content,
  style,
  className,
}: {
  title: string | null;
  content: ReactElement;
  style: { [key: string]: any } | null;
  className: string | null;
}): ReactElement {
  return (
    <Popover
      title={title}
      content={<div style={{ maxWidth: 300 }}>{content}</div>}
    >
      <QuestionCircleOutlined style={style} className={className} />
    </Popover>
  );
}

HelpIcon.defaultProps = {
  title: null,
  className: null,
  style: null,
};
