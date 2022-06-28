import React, { ReactElement, useEffect } from "react";
import { useSelector } from "react-redux";
import { useHistory } from "react-router-dom";
import { Layout, Menu, Breadcrumb, Typography, Divider } from "antd";
import type { MenuProps } from "antd";
import { createUseStyles } from "react-jss";
import {
  LaptopOutlined,
  NotificationOutlined,
  UserOutlined,
} from "@ant-design/icons";
import { RootState } from "./app/store";
import { DOCUMENTATION_ROUTE } from "./routes";
import DocumentationModal from "./components/modals/DocumentationModal";

const items1: MenuProps["items"] = ["1", "2", "3"].map((key) => ({
  key,
  label: `nav ${key}`,
}));

const items2: MenuProps["items"] = [
  UserOutlined,
  LaptopOutlined,
  NotificationOutlined,
].map((icon, index) => {
  const key = String(index + 1);

  return {
    key: `sub${key}`,
    icon: React.createElement(icon),
    label: `subnav ${key}`,

    children: new Array(4).fill(null).map((_, j) => {
      const subKey = index * 4 + j + 1;
      return {
        key: subKey,
        label: `option${subKey}`,
      };
    }),
  };
});

const useStyles = createUseStyles({
  logoWrapper: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 24,
    backgroundColor: "white",
  },
  logo: {
    color: "#161619 !important",
    marginBottom: "0px !important",
    fontSize: "20px !important",
    marginTop: 6,
    fontFamily: "atmospheric",
    backgroundColor: "white",
  },
  logoVersion: {
    color: "#161619 !important",
    fontWeight: "bold",
    fontSize: 12,
  },
});

export default function Documentation(): ReactElement {
  const page = useSelector(
    (root: RootState) => root.documentationManager.route
  );
  const history = useHistory();
  const appInfo = useSelector((state: RootState) => state.appInfo);
  const classes = useStyles();

  useEffect(() => {
    history.push(page);
  }, [page]);

  return (
    <DocumentationModal>
      <Layout>
        <Layout.Sider width={200} style={{ backgroundColor: "#fff" }}>
          <div className={classes.logoWrapper}>
            <Typography.Title level={3} className={classes.logo}>
              VOICESMITH
            </Typography.Title>
            <Typography.Text className={classes.logoVersion}>
              {`Documentation ${appInfo === null ? "" : `v${appInfo.version}`}`}
            </Typography.Text>
          </div>
          <div style={{ paddingLeft: 8, paddingRight: 8 }}>
            <Divider />
            <Menu
              mode="inline"
              defaultSelectedKeys={["1"]}
              defaultOpenKeys={["sub1"]}
              style={{ height: "100%", borderRight: 0 }}
              items={items2}
            />
          </div>
        </Layout.Sider>
        <Layout style={{ padding: "0 24px 24px" }}>
          <Breadcrumb style={{ margin: "16px 0" }}>
            <Breadcrumb.Item>Home</Breadcrumb.Item>
            <Breadcrumb.Item>List</Breadcrumb.Item>
            <Breadcrumb.Item>App</Breadcrumb.Item>
          </Breadcrumb>
          <Layout.Content
            style={{
              padding: 24,
              margin: 0,
              minHeight: 280,
            }}
          >
            Content
          </Layout.Content>
        </Layout>
      </Layout>
    </DocumentationModal>
  );
}
