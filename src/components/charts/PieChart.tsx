import React, { ReactElement } from "react";
import { Card } from "antd";
import { createUseStyles } from "react-jss";
import { RadialChart } from "react-vis";
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
  chartHeight: number;
  chartWidth: number;
}): ReactElement {
  const classes = useStyles();
  const radialData = data.map((el: number, index: number) => ({
    angle: el,
    label: labels[index],
    color: CHART_BG_COLORS[index],
  }));
  return (
    <Card title={title} className={classes.card}>
      <div className={classes.cardInner}>
        <RadialChart
          data={radialData}
          height={chartHeight}
          width={chartWidth}
          colorType="literal"
        ></RadialChart>
      </div>
    </Card>
  );
}
