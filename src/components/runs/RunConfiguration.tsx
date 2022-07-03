import React, { ReactElement } from "react";
import { Button } from "antd";
import RunConfigurationForm from "../../components/runs/RunConfigurationForm";
import RunCard from "../../components/cards/RunCard";

export default function RunConfiguration({
  title,
  forms,
  hasStarted,
  isDisabled,
  onBack,
  onDefaults,
  onSave,
  onNext,
  formRef,
  initialValues,
  onFinish,
  fetchNames,
  docsUrl,
}: {
  title: string;
  forms: ReactElement;
  hasStarted: boolean;
  isDisabled: boolean;
  onBack: () => void;
  onDefaults: () => void;
  onSave: () => void;
  onNext: () => void;
  formRef: any;
  initialValues: { [key: string]: any };
  onFinish: (values: any) => void;
  fetchNames: () => Promise<string[]>;
  docsUrl: string | null;
}): ReactElement {
  return (
    <RunCard
      title={title}
      docsUrl={docsUrl}
      buttons={[
        <Button onClick={onBack}>Back</Button>,
        <Button disabled={isDisabled} onClick={onDefaults}>
          Reset to Default
        </Button>,
        <Button disabled={isDisabled} onClick={onSave}>
          Save
        </Button>,
        <Button type="primary" disabled={isDisabled} onClick={onNext}>
          {hasStarted ? "Save and Next" : "Save and Start Run"}
        </Button>,
      ]}
    >
      <RunConfigurationForm
        formRef={formRef}
        initialValues={initialValues}
        onFinish={onFinish}
        fetchNames={fetchNames}
        isDisabled={isDisabled}
      >
        {forms}
      </RunConfigurationForm>
    </RunCard>
  );
}

RunConfiguration.defaultProps = {
  docsUrl: null,
};
