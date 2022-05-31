import React, { ReactElement, useState } from "react";
import { Card, Empty } from "antd";
import { createUseStyles } from "react-jss";
import {
  HorizontalGridLines,
  XAxis,
  YAxis,
  LineSeries,
  FlexibleWidthXYPlot,
  DiscreteColorLegend,
  Crosshair,
  AreaSeries,
} from "react-vis";
import { CHART_BG_COLORS, CHART_BG_COLORS_FADED } from "../../config";
import { GraphStatisticInterface } from "../../interfaces";

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
  xLabel,
  yLabel,
  lines,
  labels,
  chartHeight,
  displayXAxis,
  roundToDecimals,
  withArea,
}: {
  title: string;
  xLabel: string | null;
  yLabel: string | null;
  lines: Array<GraphStatisticInterface[]>;
  labels: string[];
  chartHeight: number;
  displayXAxis: boolean;
  roundToDecimals: number;
  withArea: boolean;
}): ReactElement {
  const classes = useStyles();
  const [crosshairValues, setCrosshairValues] = useState([]);
  const [crosshairIsVisible, setCrosshairIsVisible] = useState(false);

  let max = -Infinity;
  const linesMapped = lines.map((statistics: GraphStatisticInterface[]) =>
    statistics.map((statistic: GraphStatisticInterface) => {
      if (statistic.value > max) {
        max = statistic.value;
      }
      return {
        x: statistic.step,
        y: parseFloat(statistic.value.toFixed(roundToDecimals)),
      };
    })
  );

  if (max === -Infinity) {
    max = 1.0;
  }

  const onNearestX = (value: any, { index }: { index: number }) => {
    setCrosshairValues(linesMapped.map((d) => d[index]));
  };

  const onMouseEnter = () => {
    setCrosshairIsVisible(true);
  };

  const onMouseLeave = () => {
    setCrosshairIsVisible(false);
  };

  const withLegend = labels.length > 1;

  return (
    <Card title={title} className={classes.card}>
      <div
        className={classes.cardInner}
        style={{ paddingBottom: withLegend ? 40 : null }}
      >
        {lines[0].length === 0 ? (
          <Empty
            description="No data received yet"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <FlexibleWidthXYPlot
            height={chartHeight}
            yDomain={[0, max]}
            onMouseLeave={onMouseLeave}
            onMouseEnter={onMouseEnter}
          >
            <HorizontalGridLines />
            {withLegend && (
              <DiscreteColorLegend
                orientation="horizontal"
                items={labels}
              ></DiscreteColorLegend>
            )}
            {crosshairIsVisible && (
              <Crosshair values={crosshairValues}></Crosshair>
            )}
            {displayXAxis && <XAxis title={xLabel} orientation="bottom" />}
            <YAxis title={yLabel} />
            {linesMapped.map((line, index) =>
              withArea ? (
                <AreaSeries
                  data={line}
                  key={index}
                  onNearestX={index === 0 ? onNearestX : null}
                />
              ) : (
                <LineSeries
                  data={line}
                  key={index}
                  onNearestX={index === 0 ? onNearestX : null}
                />
              )
            )}
          </FlexibleWidthXYPlot>
        )}
      </div>
    </Card>
  );
}

LineChart.defaultProps = {
  xLabel: null,
  yLabel: null,
  displayXAxis: true,
  labels: [],
  roundToDecimals: 4,
  withArea: false,
};

export default LineChart;
