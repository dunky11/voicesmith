import React, { ReactElement } from "react";
import { Breadcrumb, Row, Col } from "antd";
import { useDispatch, useSelector } from "react-redux";
import { RootState } from "../../app/store";
import RunCard from "../../components/cards/RunCard";
import { createUseStyles } from "react-jss";

const useStyles = createUseStyles({
  breadcrumb: { marginBottom: 8 },
});

export default function RunQueue(): ReactElement {
  const dispatch = useDispatch();
  const runManager = useSelector((state: RootState) => {
    state.runManager;
  });
  const classes = useStyles();

  return (
    <>
      <Breadcrumb className={classes.breadcrumb}>
        <Breadcrumb.Item>Run Queue</Breadcrumb.Item>
      </Breadcrumb>
      <Row>
        <Col span={12}>
          <RunCard title="Run Queue">RunQueue</RunCard>
        </Col>
      </Row>
    </>
  );
}
