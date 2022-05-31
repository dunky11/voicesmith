import React, { ReactElement } from "react";
import { Card } from "antd";
import { createUseStyles } from "react-jss";
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
}): ReactElement {
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
        ></div>
      </div>
    </Card>
  );
}
