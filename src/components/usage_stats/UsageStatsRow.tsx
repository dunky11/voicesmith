import React, { ReactElement, useEffect, useRef } from "react";
import { Row, Col } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import { addStats } from "../../features/usageStatsSlice";
import { UsageStatsInterface } from "../../interfaces";
import PieChart from "../charts/PieChart";
import LineChart from "../charts/LineChart";
import { STATISTIC_HEIGHT, SERVER_URL } from "../../config";
import { useInterval } from "../../utils";

export default function UsageStatsRow({
  style,
}: {
  style: Record<string, unknown>;
}): ReactElement {
  const usageStats = useSelector((state: RootState) => state.useStats);
  const dispatch = useDispatch();
  const isMounted = useRef(false);
  const cpuUsages = usageStats.map((el) => el.cpuUsage);
  const totalRams = usageStats.map((el) => el.totalRam);
  const ramsUsed = usageStats.map((el) => el.ramUsed);
  const diskUsed =
    usageStats.length === 0
      ? undefined
      : usageStats[usageStats.length - 1].diskUsed;
  const diskFree =
    usageStats.length === 0
      ? undefined
      : usageStats[usageStats.length - 1].totalDisk -
        (diskUsed === undefined ? 0 : diskUsed);

  const getRAMUsageTitle = () => {
    if (ramsUsed.length === 0 || totalRams.length === 0) {
      return "RAM Usage (GB)";
    }
    return `RAM Usage (${ramsUsed[ramsUsed.length - 1]}GB / ${
      totalRams[totalRams.length - 1]
    }GB)`;
  };

  const getCPUUsageTitle = () => {
    if (cpuUsages.length === 0) {
      return "CPU Usage (%)";
    }
    return `CPU Usage (${cpuUsages[cpuUsages.length - 1]}%)`;
  };

  const getDiskUsageTitle = () => {
    if (diskUsed === undefined || diskFree === undefined) {
      return "Disk Usage (GB)";
    }
    return `Disk Usage (${diskUsed}GB / ${diskUsed + diskFree}GB)`;
  };

  const pollUsageInfo = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", `${SERVER_URL}/get-system-info`);
    ajax.onload = () => {
      if (!isMounted.current) {
        return;
      }
      const response: UsageStatsInterface = JSON.parse(ajax.responseText);
      dispatch(
        addStats({
          cpuUsage: response["cpuUsage"],
          diskUsed: parseFloat(response["diskUsed"].toFixed(2)),
          totalDisk: parseFloat(response["totalDisk"].toFixed(2)),
          ramUsed: parseFloat(response["ramUsed"].toFixed(2)),
          totalRam: parseFloat(response["totalRam"].toFixed(2)),
        })
      );
    };
    ajax.send();
  };

  useInterval(pollUsageInfo, 1000);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  });

  return (
    <div style={style}>
      <Row gutter={[16, 100]}>
        <Col span={8}>
          <LineChart
            title={getCPUUsageTitle()}
            labels={["CPU Usage (%)"]}
            chartHeight={STATISTIC_HEIGHT}
            lines={[
              cpuUsages.map((el, index) => ({
                step: index,
                value: el,
                name: "",
              })),
            ]}
            animated
            withArea
          ></LineChart>
        </Col>
        <Col className="gutter-row" span={8}>
          <LineChart
            title={getRAMUsageTitle()}
            labels={["RAM Usage (GB)"]}
            chartHeight={STATISTIC_HEIGHT}
            lines={[
              ramsUsed.map((el, index) => ({
                step: index,
                value: el,
                name: "",
              })),
            ]}
            animated
            withArea
          ></LineChart>
        </Col>
        <Col className="gutter-row" span={8}>
          <PieChart
            labels={["Space Used", "Space Available"]}
            data={
              diskUsed === undefined || diskFree === undefined
                ? []
                : [diskUsed, diskFree]
            }
            title={getDiskUsageTitle()}
            chartHeight={150}
            chartWidth={150}
          ></PieChart>
        </Col>
      </Row>
    </div>
  );
}

UsageStatsRow.defaultProps = {
  style: undefined,
};
