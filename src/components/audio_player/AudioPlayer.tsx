import React, { ReactElement, useEffect, useRef } from "react";
// @ts-ignore
import WaveSurfer from "wavesurfer.js";
import { createUseStyles } from "react-jss";
import { WaveSurverInterface } from "../../interfaces";
import { GET_AUDIO_DATA_URL_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");

const useStyles = createUseStyles({
  waveWrapper: {
    display: "block",
    width: "100%",
  },
});

export default function AudioPlayer({
  id,
  path,
  onPlayStateChange,
  isPlaying,
  height,
}: {
  id: string;
  path: string;
  onPlayStateChange: (state: boolean) => void;
  isPlaying: boolean;
  height: number;
}): ReactElement {
  const classes = useStyles();
  const isMounted = useRef(false);
  const wavesurfer = useRef<WaveSurverInterface | null>(null);
  const initWaveSurfer = () => {
    wavesurfer.current = WaveSurfer.create({
      container: `#${id}`,
      waveColor: "grey",
      progressColor: "#1890ff",
      cursorColor: "#1890ff",
      barWidth: 1,
      cursorWidth: 1,
      height: height,
      barGap: 1,
    });
    if (wavesurfer.current != null) {
      wavesurfer.current.on("finish", stopAudio);
    }
  };

  const getDataUrl = () => {
    if (path === null) {
      return;
    }
    ipcRenderer
      .invoke(GET_AUDIO_DATA_URL_CHANNEL.IN, path)
      .then((dataUrl: string) => {
        if (wavesurfer.current === null || !isMounted.current) {
          return;
        }
        wavesurfer.current.load(dataUrl);
      });
  };

  const stopAudio = () => {
    if (wavesurfer.current === null) {
      return;
    }
    wavesurfer.current.stop();
    onPlayStateChange(false);
  };

  useEffect(() => {
    getDataUrl();
  }, [path]);

  useEffect(() => {
    if (wavesurfer.current === null) {
      return;
    }
    if (!isPlaying) {
      wavesurfer.current.pause();
    } else {
      wavesurfer.current.play();
    }
  }, [isPlaying]);

  useEffect(() => {
    isMounted.current = true;
    initWaveSurfer();
    getDataUrl();
    return () => {
      isMounted.current = false;
    };
  }, []);

  return <div id={id} className={classes.waveWrapper}></div>;
}

AudioPlayer.defaultProps = {
  height: 80,
};
