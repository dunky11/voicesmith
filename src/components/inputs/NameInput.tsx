import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Form, Input } from "antd";

export default function DeviceInput({
  disabled,
  fetchNames,
}: {
  disabled: boolean;
  fetchNames: () => Promise<string[]>;
}): ReactElement {
  const [names, setNames] = useState<string[]>([]);
  const isMounted = useRef(false);

  const fetchNamesInUse = async () => {
    const names = await fetchNames();
    if (isMounted.current) {
      setNames(names);
    }
  };

  useEffect(() => {
    isMounted.current = true;
    fetchNamesInUse();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return (
    <Form.Item
      label="Name"
      name="name"
      rules={[
        () => ({
          validator(_, value: string) {
            if (value.trim() === "") {
              return Promise.reject(new Error("Please enter a name"));
            }
            if (names.includes(value)) {
              return Promise.reject(new Error("This name is already in use"));
            }
            return Promise.resolve();
          },
        }),
      ]}
    >
      <Input disabled={disabled}></Input>
    </Form.Item>
  );
}

DeviceInput.defaultProps = {
  disabled: false,
};
