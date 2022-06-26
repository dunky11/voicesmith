import React, { ReactElement } from "react";
import { Breadcrumb, Typography } from "antd";
import { Link } from "react-router-dom";
import { useSelector } from "react-redux";
import { RootState } from "../../app/store";

export default function BreadcrumbItem({
  to,
  children,
}: {
  to: string | null;
  children: string;
}): ReactElement {
  const isDisabled = useSelector(
    (state: RootState) => state.navigationSettings.isDisabled
  );
  return (
    <Breadcrumb.Item>
      {to === null ? (
        <Typography.Text>{children}</Typography.Text>
      ) : isDisabled ? (
        <Typography.Text disabled={isDisabled}>{children}</Typography.Text>
      ) : (
        <Link to={to}>{children}</Link>
      )}
    </Breadcrumb.Item>
  );
}

BreadcrumbItem.defaultProps = {
  onClick: null,
  disabled: false,
  to: null,
};
