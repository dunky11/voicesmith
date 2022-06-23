import React, { ReactElement } from "react";
import { Breadcrumb, Table, Space, Typography, Button, Tag } from "antd";
import { createUseStyles } from "react-jss";
import { SyncOutlined } from "@ant-design/icons";
import { useDispatch, useSelector } from "react-redux";
import { getStateTag, getTypeTag } from "../../utils";
import { RootState } from "../../app/store";
import RunCard from "../../components/cards/RunCard";
import { defaultPageOptions } from "../../config";
import { editQueue, setIsRunning } from "../../features/runManagerSlice";

const useStyles = createUseStyles({
  breadcrumb: { marginBottom: 8 },
  toggleRunButton: { marginBottom: 16 },
});

export default function RunQueue(): ReactElement {
  const dispatch = useDispatch();
  const runManager = useSelector((state: RootState) => state.runManager);
  const classes = useStyles();

  const onStartRunClick = () => {
    dispatch(setIsRunning(true));
  };

  const onPauseRunClick = () => {
    dispatch(setIsRunning(false));
  };

  const removeFromQueue = (position: number) => {
    const newQueue = [...runManager.queue];
    newQueue.splice(position, 1);
    dispatch(editQueue(newQueue));
  };

  const moveUp = (position: number) => {
    const newQueue = [...runManager.queue];
    newQueue[position - 1] = runManager.queue[position];
    newQueue[position] = runManager.queue[position - 1];
    dispatch(editQueue(newQueue));
  };

  const moveDown = (position: number) => {
    const newQueue = [...runManager.queue];
    newQueue[position + 1] = runManager.queue[position];
    newQueue[position] = runManager.queue[position + 1];
    dispatch(editQueue(newQueue));
  };

  const columns = [
    {
      title: "State",
      key: "state",
      render: (text: any, record: any) =>
        getStateTag(record, runManager.isRunning, runManager.queue),
    },
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "Type",
      key: "type",
      render: (text: any, record: any) => getTypeTag(record.type),
    },
    {
      title: "",
      key: "action",
      render: (text: any, record: any) => {
        const isFirst = record.position === 0;
        const isLast = record.position === runManager.queue.length - 1;
        const disableBecauseFirst = isFirst && runManager.isRunning;
        const moveUpDisabled =
          isFirst || (record.position === 1 && runManager.isRunning);
        const moveDownDisabled = isLast || disableBecauseFirst;
        return (
          <Space size="middle">
            {disableBecauseFirst ? (
              <Typography.Text disabled>Remove From Queue</Typography.Text>
            ) : (
              <a
                onClick={() => {
                  removeFromQueue(record.position);
                }}
              >
                Remove From Queue
              </a>
            )}
            {moveUpDisabled ? (
              <Typography.Text disabled>Move Up</Typography.Text>
            ) : (
              <a
                onClick={() => {
                  moveUp(record.position);
                }}
              >
                Move Up
              </a>
            )}
            {moveDownDisabled ? (
              <Typography.Text disabled>Move Down</Typography.Text>
            ) : (
              <a
                onClick={() => {
                  moveDown(record.position);
                }}
              >
                Move Down
              </a>
            )}
          </Space>
        );
      },
    },
  ];

  return (
    <>
      <Breadcrumb className={classes.breadcrumb}>
        <Breadcrumb.Item>Run Queue</Breadcrumb.Item>
      </Breadcrumb>
      <RunCard title="Run Queue" disableFullHeight>
        <Button
          disabled={runManager.queue.length === 0}
          onClick={runManager.isRunning ? onPauseRunClick : onStartRunClick}
          className={classes.toggleRunButton}
        >
          {runManager.isRunning ? "Pause Run" : "Start Run"}
        </Button>
        <Table
          bordered
          style={{ width: "100%" }}
          columns={columns}
          pagination={defaultPageOptions}
          dataSource={runManager.queue.map((run, index) => ({
            ...run,
            position: index,
            key: `${run.type}-${run.ID}`,
          }))}
        ></Table>
      </RunCard>
    </>
  );
}
