import React, { useState, useEffect, useRef } from "react";
import { Typography } from "antd";
import { POLL_LOGFILE_INTERVALL } from "../../config";
import { useInterval } from "../../utils";
import { createUseStyles } from "react-jss";
import Terminal from "./Terminal";
const { ipcRenderer } = window.require("electron");
const useStyles = createUseStyles({
  wrapper: {
    maxHeight: 600,
    overflowY: "auto",
    background: "rgba(150, 150, 150, .1)",
    border: "1px solid rgba(100, 100, 100, .2)",
    padding: 16,
  },
  text: { whiteSpace: "pre-wrap" },
});

export default function LogPrinter({
  name,
  logFileName,
  type,
}: {
  name: string | null;
  logFileName: string;
  type: "trainingRun" | "model" | "cleaningRun" | "textNormalizationRun";
}) {
  const classes = useStyles();
  const [logLines, setLogLines] = useState<string[]>([]);
  const isMounted = useRef(false);

  const pollLog = () => {
    if (name === null) {
      return;
    }
    ipcRenderer
      .invoke("fetch-logfile", name, logFileName, type)
      .then((lines: string[]) => {
        if (!isMounted.current) {
          return;
        }
        if (lines.length !== logLines.length) {
          setLogLines(lines);
        }
      });
  };

  useInterval(pollLog, POLL_LOGFILE_INTERVALL);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <Terminal
      messages={logLines.map((el) => ({ type: "message", message: el }))}
    />
  );
}
