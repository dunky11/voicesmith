import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Form, Select, Typography } from "antd";
import HelpIcon from "../help/HelpIcon";
import { DatasetInterface } from "../../interfaces";
import { FETCH_DATASET_CANDIDATES_CHANNEL } from "../../channels";

const { ipcRenderer } = window.require("electron");

export default function DatasetInput({
  disabled,
  docsUrl,
}: {
  disabled: boolean;
  docsUrl: string | null;
}): ReactElement {
  const isMounted = useRef(false);
  const [datasets, setDatasets] = useState<DatasetInterface[]>([]);

  const fetchDatasets = () => {
    ipcRenderer
      .invoke(FETCH_DATASET_CANDIDATES_CHANNEL.IN)
      .then((datasets: DatasetInterface[]) => {
        console.log(datasets);
        if (!isMounted.current) {
          return;
        }
        setDatasets(datasets);
      });
  };

  useEffect(() => {
    isMounted.current = true;
    fetchDatasets();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <Form.Item
      rules={[
        () => ({
          validator(_, value: string) {
            if (value === null) {
              return Promise.reject(new Error("Please select a dataset"));
            }
            return Promise.resolve();
          },
        }),
      ]}
      label={
        <Typography.Text>
          Dataset
          {docsUrl && <HelpIcon docsUrl={docsUrl} style={{ marginLeft: 8 }} />}
        </Typography.Text>
      }
      name="datasetID"
    >
      <Select disabled={disabled}>
        {datasets.map((dataset: DatasetInterface) => (
          <Select.Option
            value={dataset.ID}
            key={dataset.ID}
            disabled={dataset.referencedBy !== null}
          >
            {dataset.name}
          </Select.Option>
        ))}
      </Select>
    </Form.Item>
  );
}

DatasetInput.defaultProps = {
  disabled: false,
  docsUrl: null,
};
