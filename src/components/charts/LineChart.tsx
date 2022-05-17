import React from "react";
import { Card, Empty } from "antd";
import { createUseStyles } from "react-jss";
import "chart.js/auto";
import { Line } from "react-chartjs-2";
import { CHART_BG_COLORS, CHART_BG_COLORS_FADED } from "../../config";

const useStyles = createUseStyles({
  card: {
    width: "100%",
  },
  cardInner: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
});

function LineChart({
  title,
  steps,
  data,
  labels,
  chartHeight,
  chartWidth,
  displayLegend,
  displayXAxis,
  displayYAxis,
  minY,
  maxY,
  withArea,
  disableAnimation,
}: {
  title: string;
  steps: number[];
  data: Array<number | null>[];
  labels: string[];
  chartHeight: number | string;
  chartWidth: number | string;
  displayLegend: boolean;
  displayXAxis: boolean;
  displayYAxis: boolean;
  minY: number | undefined;
  maxY: number | undefined;
  withArea: boolean;
  disableAnimation: boolean;
}) {
  const classes = useStyles();
  return (
    <Card title={title} className={classes.card}>
      <div className={classes.cardInner}>
        <div
          style={{
            position: "relative",
            height: chartHeight,
            width: chartWidth,
          }}
        >
          {steps.length === 0 ? (
            <Empty
              description="No data received yet"
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <Line
              data={{
                labels: steps,
                datasets: labels.map((label, index) => ({
                  label: label,
                  data: data[index],
                  backgroundColor: CHART_BG_COLORS[index],
                  borderColor: CHART_BG_COLORS_FADED[index],
                  fill: withArea
                    ? {
                        target: "origin",
                        above: CHART_BG_COLORS_FADED[index], // And blue below the origin
                      }
                    : undefined,
                })),
              }}
              options={{
                normalized: true,
                spanGaps: true,
                datasets: {
                  line: {
                    pointRadius: 0, // disable for all `'line'` datasets
                  },
                },
                elements: {
                  point: {
                    radius: 0, // default to disabled in all datasets
                  },
                },
                animation: disableAnimation ? false : undefined,
                plugins: {
                  legend: {
                    display: displayLegend,
                  },
                },
                scales: {
                  x: {
                    display: displayXAxis,
                  },
                  y: {
                    display: displayYAxis,
                    min: minY,
                    max: maxY,
                  },
                },
                interaction: {
                  mode: "index",
                  intersect: false,
                },
              }}
            />
          )}
        </div>
      </div>
    </Card>
  );
}

LineChart.defaultProps = {
  displayLegend: false,
  displayXAxis: true,
  displayYAxis: true,
  minY: 0,
  withArea: false,
  maxY: undefined,
  disableAnimation: false,
};

export default LineChart;
