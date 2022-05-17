import React from "react";
import { Typography, Card } from "antd";
import classNames from "classnames";
import { createUseStyles } from "react-jss";
import { TerminalMessage } from "../../interfaces";

const useStyles = createUseStyles({
  wrapper: {
    width: "100%",
    maxHeight: 600,
    overflowY: "auto",
    padding: 16,
    backgroundColor: "#272727",
    borderRadius: 2,
  },
  text: { fontFamily: "monospace", fontSize: 12 },
  message: {
    color: "#9CD9F0",
  },
  errorMessage: {
    color: "#E09690",
  },
});

export default function Terminal({
  messages,
  maxLines,
}: {
  messages: TerminalMessage[];
  maxLines: number;
}) {
  const classes = useStyles();
  const startIndex = Math.max(messages.length - maxLines, 0);

  return (
    <div className={classes.wrapper}>
      {messages.slice(startIndex).map((el, index) => (
        <Typography.Paragraph
          className={classNames(
            classes.text,
            el.type === "message" ? classes.message : classes.errorMessage
          )}
          key={startIndex + index}
        >
          {el.message}
        </Typography.Paragraph>
      ))}
    </div>
  );
}

Terminal.defaultProps = { maxLines: 1000 };
