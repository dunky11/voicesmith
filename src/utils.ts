import React, { useState, useCallback, useEffect, useRef } from "react";
import { notification } from "antd";
import { GraphStatisticInterface, RunInterface } from "./interfaces";

export function useInterval(callback: any, delay: number | null) {
  // FROM https://stackoverflow.com/questions/53024496/state-not-updating-when-using-react-state-hook-within-setinterval

  const intervalRef = React.useRef<number | undefined>();
  const callbackRef = React.useRef(callback);

  // Remember the latest callback:
  //
  // Without this, if you change the callback, when setInterval ticks again, it
  // will still call your old callback.
  //
  // If you add `callback` to useEffect's deps, it will work fine but the
  // interval will be reset.

  React.useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  // Set up the interval:

  React.useEffect(() => {
    if (typeof delay === "number") {
      intervalRef.current = window.setInterval(
        () => callbackRef.current(),
        delay
      );
      callbackRef.current();

      // Clear interval if the components is unmounted or the delay changes:
      return () => window.clearInterval(intervalRef.current);
    }
  }, [delay]);

  // Returns a ref to the interval ID in case you want to clear it manually:
  return intervalRef;
}

export function useStateCallback<T>(initialState: T) {
  const [state, setState] = useState(initialState);
  const cbRef = useRef<() => void | null>(null); // init mutable ref container for callbacks

  const setStateCallback = useCallback((state: T, cb: null | "() => void") => {
    // @ts-ignore
    cbRef.current = cb; // store current, passed callback in ref
    setState(state);
  }, []); // keep object reference stable, exactly like `useState`

  useEffect(() => {
    // cb.current is `null` on initial render,
    // so we only invoke callback on state *updates*
    if (cbRef.current) {
      cbRef.current();
      cbRef.current = null; // reset callback after execution
    }
  }, [state]);

  return [state, setStateCallback];
}

export function getCategoricalGraphStat(
  graphStatistics: GraphStatisticInterface[],
  trainName: string,
  valName: string
) {
  const trainStats = graphStatistics.filter((graphStatistic) => {
    return graphStatistic.name === trainName;
  });
  const valStats = graphStatistics.filter((graphStatistic) => {
    return graphStatistic.name === valName;
  });
  const steps = trainStats.map((el) => {
    return el.step;
  });
  const trainData: Array<number> = [];
  const valData: Array<number | null> = [];
  let valIndex = 0;
  trainStats.map((el, index) => {
    trainData.push(el.value);
    if (valIndex > valStats.length - 1) {
      return;
    }
    const valSample = valStats[valIndex];
    if (valSample.step <= el.step) {
      valData.push(valSample.value);
      valIndex += 1;
    } else {
      valData.push(null);
    }
  });
  const data = [trainData, valData];
  const labels = ["Training Loss", "Validation Loss"];
  return {
    steps,
    data,
    labels,
  };
}

export function generateUUID() {
  // FROM https://stackoverflow.com/questions/6860853/generate-random-string-for-div-id
  const s4 = function () {
    return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
  };
  return (
    s4() +
    s4() +
    "-" +
    s4() +
    "-" +
    s4() +
    "-" +
    s4() +
    "-" +
    s4() +
    s4() +
    s4()
  );
}

export function getProgressTitle(title: string, progress: number | null) {
  return `${title} (${(progress === null ? 0 : progress * 100).toPrecision(
    3
  )}%)`;
}

export function numberCompare(a: number, b: number) {
  return b - a;
}

export function stringCompare(a: string, b: string) {
  return a.localeCompare(b);
}

export function getStageIsRunning(
  pageStates: string[],
  runningStage: string | null,
  running: RunInterface | null,
  type: "trainingRun" | "dSCleaning" | "textNormalizationRun",
  ID: number | null
) {
  return (
    pageStates.includes(runningStage) &&
    running !== null &&
    running.type === type &&
    running.ID === ID
  );
}

export function getWouldContinueRun(
  pageStates: string[],
  runningStage: string | null,
  running: RunInterface | null,
  type: "trainingRun" | "dSCleaning" | "textNormalizationRun",
  ID: number | null
) {
  return (
    pageStates.includes(runningStage) ||
    (running !== null && (running.type !== type || running.ID !== ID))
  );
}

export const notifySave = () => {
  notification["success"]({
    message: "Your settings have been saved",
    placement: "top",
  });
};
