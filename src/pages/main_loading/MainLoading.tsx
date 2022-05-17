import React from "react";
import { Spin } from "antd";
import "./MainLoading.css";

export default function MainLoading({}: {}) {
  return (
    <div className="main-loading-wrapper">
      <Spin size="large"></Spin>
    </div>
  );
}
