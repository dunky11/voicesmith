import React, { ReactNode } from "react";
import { Card } from "antd";

export default function RunCard({
  title,
  buttons,
  children,
  disableFullHeight,
}: {
  title: string | null;
  buttons: ReactNode[];
  children: ReactNode;
  disableFullHeight: boolean;
}) {
  return (
    <Card
      title={title}
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
};
