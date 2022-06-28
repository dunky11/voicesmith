import { Typography } from "antd";
import React, { ReactElement } from "react";
import DocumentationCard from "../../components/cards/DocumentationCard";

export default function Introduction(): ReactElement {
  return (
    <DocumentationCard title="Introduction">
      <Typography.Text>Introduction</Typography.Text>
    </DocumentationCard>
  );
}
