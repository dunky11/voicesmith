import React, { ReactElement } from "react";
import { Form } from "antd";
import NameInput from "../inputs/NameInput";

export default function RunConfigurationForm({
  formRef,
  initialValues,
  onFinish,
  fetchNames,
  isDisabled,
  children,
}: {
  // TODO find correct type
  formRef: any;
  initialValues: { [key: string]: any };
  onFinish: (values: any) => void;
  fetchNames: () => Promise<string[]>;
  isDisabled: boolean;
  children: ReactElement;
}): ReactElement {
  return (
    <Form
      layout="vertical"
      ref={(node) => {
        formRef.current = node;
      }}
      initialValues={initialValues}
      onFinish={onFinish}
    >
      <NameInput disabled={isDisabled} fetchNames={fetchNames} />
      {children}
    </Form>
  );
}
