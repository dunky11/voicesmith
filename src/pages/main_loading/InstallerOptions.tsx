import React, { useEffect, useState } from "react";
import {
  Typography,
  Radio,
  Space,
  RadioChangeEvent,
  Modal,
  Checkbox,
  Button,
} from "antd";
import NoCloseModal from "../../components/modals/NoCloseModal";
import { FETCH_HAS_DOCKER_CHANNEL } from "../../channels";
import { AppInfoInterface, InstallerOptionsInterface } from "../../interfaces";
import { CheckboxChangeEvent } from "antd/lib/checkbox";
const { ipcRenderer, shell } = window.require("electron");

export default function InstallerOptions({
  open,
  appInfo,
  onFinish,
}: {
  open: boolean;
  appInfo: AppInfoInterface;
  onFinish: (options: InstallerOptionsInterface) => void;
}) {
  const [hasDocker, setHasDocker] = useState<boolean | null>(null);
  const [currentStep, setCurrentStep] = useState<
    "loading" | "deviceConfig" | "noDocker" | "install" | "containerToolkit"
  >();

  const [installerOptions, setInstallerOptions] =
    useState<InstallerOptionsInterface>({
      device: "CPU",
      dockerIsInstalled: null,
      hasInstalledNCT: false,
    });

  const onGPURadioChange = (event: RadioChangeEvent) => {
    setInstallerOptions({
      ...installerOptions,
      device: event.target.value,
    });
  };

  const onHasInstalledNCTChange = (event: CheckboxChangeEvent) => {
    setInstallerOptions({
      ...installerOptions,
      hasInstalledNCT: event.target.checked,
    });
  };

  const fetchHasDocker = () => {
    ipcRenderer
      .invoke(FETCH_HAS_DOCKER_CHANNEL.IN)
      .then((hasDocker: boolean) => {
        setHasDocker(hasDocker);
      });
  };

  const renderStep = () => {
    switch (currentStep) {
      case "loading":
        return <></>;
      case "deviceConfig":
        return (
          <>
            <Typography.Paragraph>
              This installer will guide you through the installation of
              VoiceSmith. First we must find out whether you want to use this
              software on GPU or not.
            </Typography.Paragraph>
            <Typography.Paragraph>
              GPU makes this software faster but requires some extra steps. It
              requires you to have a GPU with CUDA support. If you dont know
              whether it's supported by CUDA, navigate over to{" "}
              <Typography.Link
                onClick={() => {
                  shell.openExternal("https://developer.nvidia.com/cuda-gpus");
                }}
              >
                https://developer.nvidia.com/cuda-gpus
              </Typography.Link>{" "}
              to find out whether your GPU is in the list of CUDA supported
              GPUs.
            </Typography.Paragraph>
            <Radio.Group
              onChange={onGPURadioChange}
              value={installerOptions.device}
            >
              <Space direction="vertical">
                <Radio value="CPU">I don't want to train on GPU</Radio>
                <Radio value="GPU" disabled={appInfo.platform === "win32"}>
                  I want to train on GPU and my GPU is in the list of CUDA
                  enabled GPUs
                </Radio>
              </Space>
            </Radio.Group>
          </>
        );
      case "noDocker":
        return (
          <Typography.Paragraph>
            No installation of Docker detected on this system. Please navigate
            over to
            <Typography.Link
              onClick={() => {
                shell.openExternal("https://docs.docker.com/get-docker/");
              }}
            >
              https://docs.docker.com/get-docker/
            </Typography.Link>{" "}
            to download and install Docker. Afterwards restart this software.
          </Typography.Paragraph>
        );
      case "containerToolkit": {
        return (
          <>
            <Typography.Paragraph>
              In order to use this software on GPU you need to have the{" "}
              <Typography.Link
                onClick={() => {
                  shell.openExternal(
                    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker"
                  );
                }}
              >
                NVIDIA Container Toolkit
              </Typography.Link>{" "}
              installed. If you haven't already,{" "}
              <Typography.Link
                onClick={() => {
                  shell.openExternal(
                    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker"
                  );
                }}
              >
                click here
              </Typography.Link>{" "}
              and follow the instructions.
              <Typography>
                <Checkbox
                  onChange={onHasInstalledNCTChange}
                  checked={installerOptions.hasInstalledNCT}
                ></Checkbox>{" "}
                I have installed the NVIDIA Container Toolkit.
              </Typography>
            </Typography.Paragraph>
          </>
        );
      }
    }
  };

  const renderButtons = () => {
    switch (currentStep) {
      case "loading": {
        return null;
      }
      case "deviceConfig": {
        return [
          <Button
            type="primary"
            onClick={() => {
              if (hasDocker) {
                if (installerOptions.device === "CPU") {
                  onFinish(installerOptions);
                } else {
                  setCurrentStep("containerToolkit");
                }
              } else {
                setCurrentStep("noDocker");
              }
            }}
          >
            Next
          </Button>,
        ];
      }
      case "noDocker": {
        return [
          <Button
            onClick={() => {
              setCurrentStep("deviceConfig");
            }}
          >
            Back
          </Button>,
        ];
      }
      case "containerToolkit": {
        return [
          <Button
            onClick={() => {
              setCurrentStep("deviceConfig");
            }}
          >
            Back
          </Button>,
          <Button
            type="primary"
            onClick={() => {
              onFinish(installerOptions);
            }}
            disabled={!installerOptions.hasInstalledNCT}
          >
            Install
          </Button>,
        ];
      }
    }
  };

  useEffect(() => {
    if (appInfo !== null && hasDocker !== null) {
      setCurrentStep("deviceConfig");
    }
  }, [appInfo, hasDocker]);

  useEffect(() => {
    fetchHasDocker();
  }, []);

  return (
    <NoCloseModal
      title="Install VoiceSmith"
      visible={open}
      buttons={renderButtons()}
    >
      {renderStep()}
    </NoCloseModal>
  );
}
