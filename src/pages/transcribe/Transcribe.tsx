import { Breadcrumb } from "antd";
import React, { useEffect } from "react";
import { createUseStyles } from "react-jss";
import RunCard from "../../components/cards/RunCard";

const useStyles = createUseStyles({
  breadcrumb: {
    marginBottom: 8,
  },
});

const SAMPLES_PER_SECOND = 200;

export default function Transcribe() {
  const classes = useStyles();

  const visualizeAudio = (url: string) => {
    fetch(url)
      .then((response) => response.arrayBuffer())
      .then((arrayBuffer) => new AudioContext().decodeAudioData(arrayBuffer))
      .then((audioBuffer: AudioBuffer) => filterData(audioBuffer))
      .then((waveform: number[]) => normalizeData(waveform))
      .then((waveform: number[]) => draw(waveform));
  };

  const draw = (waveform: number[]) => {
    // set up the canvas
    const canvas = document.querySelector("#canvas") as HTMLCanvasElement;
    const dpr = window.devicePixelRatio || 1;
    const padding = 20;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = (canvas.offsetHeight + padding * 2) * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.translate(0, canvas.offsetHeight / 2 + padding); // set Y = 0 to be in the middle of the canvas

    // draw the line segments
    const width = canvas.offsetWidth / waveform.length;
    for (let i = 0; i < waveform.length; i++) {
      const x = width * i;
      let height = waveform[i] * canvas.offsetHeight - padding;
      if (height < 0) {
        height = 0;
      } else if (height > canvas.offsetHeight / 2) {
        height = height > canvas.offsetHeight / 2 ? 1 : 0;
      }
      drawLineSegment(ctx, x, height, width, i % 2 == 0);
    }
  };

  const drawLineSegment = (
    ctx: any,
    x: any,
    height: number,
    width: number,
    isEven: boolean
  ) => {
    ctx.lineWidth = 1; // how thick the line is
    ctx.strokeStyle = "#fff"; // what color our line is
    ctx.beginPath();
    height = isEven ? height : -height;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.arc(x + width / 2, height, width / 2, Math.PI, 0, isEven);
    ctx.lineTo(x + width, 0);
    ctx.stroke();
  };

  const normalizeData = (filteredData: number[]) => {
    const multiplier = Math.pow(
      filteredData.reduce((a, b) => {
        return Math.max(a, b);
      }, -Infinity),
      -1
    );
    return filteredData.map((n) => n * multiplier);
  };

  const filterData = (audioBuffer: AudioBuffer) => {
    const rawData = audioBuffer.getChannelData(0); // We only need to work with one channel of data
    const duration = audioBuffer.length / audioBuffer.sampleRate;
    const samplesNeeded = duration * SAMPLES_PER_SECOND;
    const blockSize = Math.floor(audioBuffer.length / samplesNeeded);
    const filteredData = [];
    for (let i = 0; i < rawData.length; i++) {
      const blockStart = blockSize * i; // the location of the first sample in the block
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum = sum + Math.abs(rawData[blockStart + j]); // find the sum of all the samples in the block
      }
      filteredData.push(sum / blockSize); // divide the sum by the block size to get the average
    }
    return filteredData;
  };

  useEffect(() => {
    visualizeAudio(
      "https://s3-us-west-2.amazonaws.com/s.cdpn.io/3/shoptalk-clip.mp3"
    );
  }, []);
  return (
    <>
      <Breadcrumb className={classes.breadcrumb}>
        <Breadcrumb.Item>Transcribe</Breadcrumb.Item>
      </Breadcrumb>
      <RunCard disableFullHeight>
        <canvas style={{ backgroundColor: "black" }} id="canvas"></canvas>
      </RunCard>
    </>
  );
}
