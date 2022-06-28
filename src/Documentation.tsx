import React, { ReactElement, useEffect } from "react";
import { useSelector } from "react-redux";
import { useHistory, Route, Switch } from "react-router-dom";
import { Layout, Menu, Typography, Divider } from "antd";
import { createUseStyles } from "react-jss";
import { RootState } from "./app/store";
import { DOCUMENTATION_ROUTE } from "./routes";
import DocumentationModal from "./components/modals/DocumentationModal";
import Introduction from "./pages/documentation/Introduction";

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

interface DocumentationMenuItemInterface {
  key: string;
  icon: ReactElement;
  label: string;
  render?: ReactElement;
  children?: { key: string; label: string; render: ReactElement }[];
}

const menuItems: DocumentationMenuItemInterface[] = [
  {
    key: DOCUMENTATION_ROUTE.INTODUCTION.ROUTE,
    icon: null,
    label: "Introduction",
    render: <Introduction />,
  },
  {
    key: "sub1",
    icon: null,
    label: "sub1",
    children: [
      {
        key: DOCUMENTATION_ROUTE.DATASETS.ROUTE,
        label: "World",
        render: <div>Hellp World</div>,
      },
    ],
  },
];

export default function Documentation(): ReactElement {
  const page = useSelector(
    (root: RootState) => root.documentationManager.route
  );
  const history = useHistory();
  const appInfo = useSelector((state: RootState) => state.appInfo);
  const classes = useStyles();

  const onMenuSelect = ({ key }: { key: string }) => {
    console.log("KEY CALLED");
    console.log(key);
    history.push("/introduction");
  };

  const renderMenuItems = (): ReactElement[] => {
    const out: ReactElement[] = [];
    for (let i = 0; i < menuItems.length; i++) {
      if (Object.prototype.hasOwnProperty.call(menuItems[i], "render")) {
        out.push(
          <Route
            key={menuItems[i].key}
            render={() => menuItems[i].render}
            path={menuItems[i].key}
            exact
          />
        );
      } else {
        for (let j = 0; j < menuItems[i].children.length; j++) {
          out.push(
            <Route
              key={menuItems[i].children[j].key}
              render={() => menuItems[i].children[j].render}
              path={menuItems[i].children[j].key}
              exact
            />
          );
        }
      }
    }
    return out;
  };

  /**
  useEffect(() => {
    history.push(page);
  }, [page]);
  */

  console.log(history);

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
              onSelect={onMenuSelect}
              items={menuItems}
            />
          </div>
        </Layout.Sider>
        <Layout style={{ padding: "0 24px 24px" }}>
          <Layout.Content
            style={{
              padding: 24,
              margin: 0,
              minHeight: 280,
            }}
          >
            <Typography>Test</Typography>
            <Switch>
              <Route
                render={() => <div>Introduction</div>}
                exact
                path="/introduction"
              ></Route>
              <Route render={() => <div>Test</div>} path="/"></Route>
            </Switch>
          </Layout.Content>
        </Layout>
      </Layout>
    </DocumentationModal>
  );
}
