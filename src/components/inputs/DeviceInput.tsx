import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Form, Select, Alert, Typography } from "antd";
import { SERVER_URL } from "../../config";
const { shell } = window.require("electron");

export default function DeviceInput({
  disabled,
}: {
  disabled: boolean;
}): ReactElement {
  const [cudaIsAvailable, setCudaIsAvailable] = useState(false);
  const [hasFetchedCuda, setHasFetchedCuda] = useState(false);
  const isMounted = useRef(false);

  const fetchIsCudaAvailable = () => {
    const ajax = new XMLHttpRequest();
    ajax.open("GET", `${SERVER_URL}/is-cuda-available`);
    ajax.onload = () => {
      if (!isMounted.current) {
        return;
      }
      const response = JSON.parse(ajax.responseText);
      setCudaIsAvailable(response.available);
      setHasFetchedCuda(true);
    };
    ajax.send();
  };

  useEffect(() => {
    isMounted.current = true;
    fetchIsCudaAvailable();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <>
      <Form.Item label="Device" name="device">
        <Select disabled={disabled}>
          <Select.Option value="CPU">CPU</Select.Option>
          <Select.Option value="GPU" disabled={!cudaIsAvailable}>
            GPU
          </Select.Option>
        </Select>
      </Form.Item>
      {hasFetchedCuda && !cudaIsAvailable && (
        <Alert
          style={{ marginBottom: 24 }}
          message={
            <Typography.Text>
              No CUDA supported GPU was detected. While you can train on CPU,
              training on GPU is highly recommended since training on CPU will
              most likely take days. If you want to train on GPU{" "}
              <a
                onClick={() => {
                  shell.openExternal("https://developer.nvidia.com/cuda-gpus");
                }}
              >
                please make sure it has CUDA support
              </a>{" "}
              and it&apos;s driver is up to date. Afterwards restart the app.
            </Typography.Text>
          }
          type="warning"
        />
      )}
    </>
  );
}

DeviceInput.defaultProps = {
  disabled: false,
};
