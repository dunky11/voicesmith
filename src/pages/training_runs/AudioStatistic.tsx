import React, { useState, useRef, useEffect, ReactElement } from "react";
import { Slider, Card, Empty, Form } from "antd";
import AudioPlayer from "../../components/audio_player/AudioPlayer";
import {
  PlayCircleTwoTone,
  PauseCircleTwoTone,
  SaveTwoTone,
} from "@ant-design/icons";
import { FormInstance } from "rc-field-form";
import { STATISTIC_HEIGHT } from "../../config";
import "./AudioStatistic.css";
import { EXPORT_FILE_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");

export default function AudioStatistics({
  name,
  steps,
  paths,
}: {
  name: string;
  steps: number[];
  paths: string[];
}): ReactElement {
  const [selectedPath, setSelectedPath] = useState<string>("");
  const [isPlaying, setIsPlaying] = useState(false);
  const id = useRef(`id-${Math.random().toString(36).substr(2, 5)}`);
  const formRef = useRef<FormInstance | null>();

  const onStepChange = (selectedStep: number) => {
    let i = 0;
    for (const step of steps) {
      if (step === selectedStep) {
        setSelectedPath(paths[i]);
        setIsPlaying(false);
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

  const startPlaying = () => {
    setIsPlaying(true);
  };

  const pausePlaying = () => {
    setIsPlaying(false);
  };

  const exportFile = () => {
    ipcRenderer.invoke(EXPORT_FILE_CHANNEL.IN, selectedPath);
  };

  useEffect(() => {
    if (paths.length > 0 && selectedPath === "") {
      setSelectedPath(paths[paths.length - 1]);
      formRef.current?.setFieldsValue({ step: steps[steps.length - 1] });
    }
  }, [paths]);

  return (
    <Card
      title={name}
      actions={
        selectedPath === ""
          ? []
          : [
              isPlaying ? (
                <PauseCircleTwoTone
                  key="pause"
                  style={{ fontSize: "150%" }}
                  onClick={pausePlaying}
                />
              ) : (
                <PlayCircleTwoTone
                  key="play"
                  style={{ fontSize: "150%" }}
                  onClick={startPlaying}
                />
              ),
              <SaveTwoTone
                key="save"
                style={{ fontSize: "150%" }}
                onClick={exportFile}
              />,
            ]
      }
    >
      {selectedPath === "" ? (
        <Empty
          description="No data received yet"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <AudioPlayer
          onPlayStateChange={setIsPlaying}
          path={selectedPath}
          isPlaying={isPlaying}
          id={id.current}
          height={STATISTIC_HEIGHT}
        ></AudioPlayer>
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
            step={null}
            disabled={selectedPath === ""}
            min={steps[0]}
            max={steps[steps.length - 1]}
            marks={statisticToMarks()}
            className="audio-slider"
          ></Slider>
        </Form.Item>
      </Form>
    </Card>
  );
}
