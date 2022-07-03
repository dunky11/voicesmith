import React, { ReactElement, ReactNode } from "react";
import { Card } from "antd";
import HelpIcon from "../help/HelpIcon";

export default function RunCard({
  title,
  buttons,
  children,
  disableFullHeight,
  docsUrl,
}: {
  title: string | null;
  buttons: ReactNode[];
  children: ReactNode;
  disableFullHeight: boolean;
  docsUrl: string | null;
}): ReactElement {
  return (
    <Card
      title={
        title && (
          <span>
            {title}
            {docsUrl && (
              <HelpIcon style={{ marginLeft: 8 }} docsUrl={docsUrl} />
            )}
          </span>
        )
      }
      style={{
        height: disableFullHeight ? null : "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
      }}
      bodyStyle={{ paddingTop: title === null ? 8 : null }}
      actions={
        buttons.length === null
          ? null
          : [
              <div
                key="next-button-wrapper"
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  marginRight: 24,
                }}
              >
                {buttons.map((ButtonNode, index) => (
                  <div
                    key={index}
                    style={{
                      marginRight: index === buttons.length - 1 ? null : 8,
                    }}
                  >
                    {ButtonNode}
                  </div>
                ))}
              </div>,
            ]
      }
    >
      {children}
    </Card>
  );
}

RunCard.defaultProps = {
  title: null,
  buttons: [],
  disableFullHeight: false,
  docsUrl: null,
};
