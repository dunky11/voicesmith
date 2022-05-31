import React, { useState, useEffect, useRef, ReactElement } from "react";
import { FETCH_LOGFILE_CHANNEL } from "../../channels";
import { POLL_LOGFILE_INTERVALL } from "../../config";
import { useInterval } from "../../utils";
import Terminal from "./Terminal";
const { ipcRenderer } = window.require("electron");

export default function LogPrinter({
  name,
  logFileName,
  type,
}: {
  name: string | null;
  logFileName: string;
  type: "trainingRun" | "model" | "cleaningRun" | "textNormalizationRun";
}): ReactElement {
  const [logLines, setLogLines] = useState<string[]>([]);
  const isMounted = useRef(false);

  const pollLog = () => {
    if (name === null) {
      return;
    }
    ipcRenderer
      .invoke(FETCH_LOGFILE_CHANNEL.IN, name, logFileName, type)
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
