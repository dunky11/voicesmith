import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  ReactElement,
} from "react";
import { InputRef, notification, Tag, Input, Space, Button } from "antd";
import { SyncOutlined, SearchOutlined } from "@ant-design/icons";
import type { FilterConfirmProps } from "antd/lib/table/interface";
import type { ColumnType } from "antd/lib/table";
import { RunInterface } from "./interfaces";
import { SERVER_URL, LANGUAGES } from "./config";

export function useInterval(
  callback: any,
  delay: number | null
): React.MutableRefObject<number | undefined> {
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

export function useStateCallback<T>(
  initialState: T
): [T, (state: T, cb: null | (() => void)) => void] {
  const [state, setState] = useState(initialState);
  const cbRef = useRef<(() => void) | null>(null); // init mutable ref container for callbacks

  const setStateCallback = useCallback((state: T, cb: null | (() => void)) => {
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

export function generateUUID(): string {
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

export function getProgressTitle(
  title: string,
  progress: number | null
): string {
  return `${title} (${(progress === null ? 0 : progress * 100).toPrecision(
    3
  )}%)`;
}

export function numberCompare(a: number, b: number): number {
  return b - a;
}

export function stringCompare(a: string, b: string): number {
  return a.localeCompare(b);
}

export function getStageIsRunning(
  pageStates: string[],
  runningStage: string | null,
  running: RunInterface | null,
  type:
    | "trainingRun"
    | "cleaningRun"
    | "textNormalizationRun"
    | "sampleSplittingRun",
  ID: number | null
): boolean {
  if (runningStage === null) {
    return false;
  }
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
  type:
    | "trainingRun"
    | "cleaningRun"
    | "textNormalizationRun"
    | "sampleSplittingRun",
  ID: number | null
): boolean {
  return (
    // @ts-ignore:
    pageStates.includes(runningStage) ||
    (running !== null && (running.type !== type || running.ID !== ID))
  );
}

export const notifySave = (): void => {
  notification["success"]({
    message: "Your settings have been saved",
    placement: "top",
  });
};

export const pingServer = (
  retryOnError: boolean,
  onSuccess: () => void
): void => {
  const ajax = new XMLHttpRequest();
  ajax.open("GET", SERVER_URL);
  ajax.onload = onSuccess;
  ajax.onerror = () => {
    if (retryOnError) {
      setTimeout(() => {
        pingServer(retryOnError, onSuccess);
      }, 50);
    }
  };
  ajax.send();
};

export const getSearchableColumn = (
  column: { [key: string]: any },
  dataIndex: string,
  searchInput: any
): { [key: string]: any } => {
  const handleSearch = (
    selectedKeys: string[],
    confirm: (param?: FilterConfirmProps) => void,
    dataIndex: string
  ) => {
    confirm();
  };

  const handleReset = (clearFilters: () => void) => {
    clearFilters();
  };

  const getColumnSearchProps = (dataIndex: string): ColumnType<any> => ({
    filterDropdown: ({
      setSelectedKeys,
      selectedKeys,
      confirm,
      clearFilters,
    }) => (
      <div style={{ padding: 8 }}>
        <Input
          ref={searchInput}
          placeholder={`Search ${dataIndex}`}
          value={selectedKeys[0]}
          onChange={(e) =>
            setSelectedKeys(e.target.value ? [e.target.value] : [])
          }
          onPressEnter={() =>
            handleSearch(selectedKeys as string[], confirm, dataIndex)
          }
          style={{ marginBottom: 8, display: "block" }}
        />
        <Space>
          <Button
            type="primary"
            onClick={() =>
              handleSearch(selectedKeys as string[], confirm, dataIndex)
            }
            icon={<SearchOutlined />}
            size="small"
            style={{ width: 90 }}
          >
            Search
          </Button>
          <Button
            onClick={() => {
              setSelectedKeys([]);
              handleReset(clearFilters);
              handleSearch([], confirm, dataIndex);
            }}
            size="small"
          >
            Reset
          </Button>
        </Space>
      </div>
    ),
    filterIcon: (filtered: boolean) => (
      <SearchOutlined style={{ color: filtered ? "#1890ff" : undefined }} />
    ),
    onFilter: (value, record) =>
      record[dataIndex]
        .toString()
        .toLowerCase()
        .includes((value as string).toLowerCase()),
    onFilterDropdownVisibleChange: (visible) => {
      if (visible) {
        setTimeout(() => searchInput.current?.select(), 100);
      }
    },
  });
  return {
    ...column,
    ...getColumnSearchProps(dataIndex),
  };
};

export const getStateTag = (
  run: RunInterface,
  isRunning: boolean,
  queue: RunInterface[]
): ReactElement => {
  let position = -1;
  for (let i = 0; i < queue.length; i++) {
    if (run.type === queue[i].type && run.ID === queue[i].ID) {
      position = i + 1;
      break;
    }
  }
  if (position === 1 && isRunning) {
    return (
      <Tag icon={<SyncOutlined spin />} color="green">
        Running
      </Tag>
    );
  } else if (position !== -1) {
    return <Tag color="blue">{`Position ${position} in queue`}</Tag>;
  } else {
    return <Tag color="orange">Not Running</Tag>;
  }
};

export const getTypeFullName = (type: RunInterface["type"]): string => {
  switch (type) {
    case "textNormalizationRun":
      return "Text Normalization";
    case "cleaningRun":
      return "Dataset Cleaning";
    case "sampleSplittingRun":
      return "Sample Splitting";
    case "trainingRun":
      return "Training Run";
    default:
      throw new Error(
        `No case selected in switch-statement, '${type}' is not a valid case ...`
      );
  }
};

export const getTypeTag = (type: RunInterface["type"]): ReactElement => {
  const name = getTypeFullName(type);
  switch (type) {
    case "textNormalizationRun":
      return <Tag>{name}</Tag>;
    case "cleaningRun":
      return <Tag>{name}</Tag>;
    case "sampleSplittingRun":
      return <Tag>{name}</Tag>;
    case "trainingRun":
      return <Tag>{name}</Tag>;
    default:
      throw new Error(
        `No case selected in switch-statement, '${type}' is not a valid case ...`
      );
  }
};

export const isInQueue = (
  run: RunInterface,
  queue: RunInterface[]
): boolean => {
  for (let i = 0; i < queue.length; i++) {
    if (run.type === queue[i].type && run.ID === queue[i].ID) {
      return true;
    }
  }
  return false;
};

const ISO6391_TO_NAME: { [key: string]: string } = {};
LANGUAGES.forEach((lang) => {
  ISO6391_TO_NAME[lang.iso6391] = lang.name;
});
export { ISO6391_TO_NAME };
