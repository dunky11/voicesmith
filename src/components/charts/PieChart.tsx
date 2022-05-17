import React from "react";
import { Card } from "antd";
import { createUseStyles } from "react-jss";
import "chart.js/auto";
import { Doughnut } from "react-chartjs-2";
import { CHART_BG_COLORS } from "../../config";

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

export default function PieChart({
  labels,
  data,
  title,
  chartHeight,
  chartWidth,
}: {
  labels: string[];
  data: number[];
  title: string;
  chartHeight: number | string;
  chartWidth: number | string;
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
          <Doughnut
            data={{
              labels: labels,
              datasets: [
                {
                  label: "My First Dataset",
                  data: data,
                  backgroundColor: CHART_BG_COLORS.slice(0, data.length),
                  hoverOffset: 4,
                },
              ],
            }}
            options={{
              plugins: {
                legend: {
                  display: false,
                },
              },
            }}
          />
        </div>
      </div>
    </Card>
  );
}
