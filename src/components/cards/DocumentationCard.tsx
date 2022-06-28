import React, { ReactElement } from "react";
import { Card } from "antd";

export default function DocumentationCard({
  title,
  children,
}: {
  title: string;
  children: ReactElement;
}): ReactElement {
  return <Card title={title}>{children}</Card>;
}
