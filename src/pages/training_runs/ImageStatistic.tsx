import React, { useState, useEffect, useRef } from "react";
import { Slider, Card, Empty, Form } from "antd";
import { STATISTIC_HEIGHT } from "../../config";
import "./ImageStatistic.css";
import Image from "../../components/image/Image";
import { FormInstance } from "rc-field-form";

export default function ImageStatistic({
  name,
  steps,
  paths,
}: {
  name: string;
  steps: number[];
  paths: string[];
}) {
  const [selectedPath, setSelectedPath] = useState("");
  const formRef = useRef<FormInstance | null>();

  const onStepChange = (selectedStep: number) => {
    let i = 0;
    for (const step of steps) {
      if (step === selectedStep) {
        setSelectedPath(paths[i]);
        return;
      }
      i += 1;
    }
  };

  const statisticToMarks = () => {
    const obj: { [step: number]: string } = {};
    for (const step of steps) {
      obj[step] = String(step);
    }
    return obj;
  };

  useEffect(() => {
    if (paths.length > 0 && selectedPath === "") {
      setSelectedPath(paths[paths.length - 1]);
      formRef.current?.setFieldsValue({ step: steps[steps.length - 1] });
    }
  }, [paths]);

  return (
    <Card title={name}>
      {selectedPath === "" ? (
        <Empty
          description="No data received yet"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <div style={{ display: "flex", justifyContent: "center" }}>
          <Image
            path={selectedPath}
            style={{
              width: "auto",
              height: STATISTIC_HEIGHT,
              maxWidth: "100%",
            }}
          ></Image>
        </div>
      )}
      <Form
        ref={(node) => {
          formRef.current = node;
        }}
        onValuesChange={(_, values) => {
          onStepChange(values.step);
        }}
      >
        <Form.Item name="step">
          <Slider
            disabled={selectedPath === ""}
            step={null}
            min={steps[0]}
            max={steps[steps.length - 1]}
            marks={statisticToMarks()}
            className="image-slider"
          ></Slider>
        </Form.Item>
      </Form>
    </Card>
  );
}
