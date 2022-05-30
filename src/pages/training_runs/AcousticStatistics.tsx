import React, { ReactElement } from "react";
import { Card, Row, Col } from "antd";
import {
  AudioStatisticInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
} from "../../interfaces";
import AudioStatistic from "./AudioStatistic";
import ImageStatistic from "./ImageStatistic";
import { getCategoricalGraphStat } from "../../utils";
import LineChart from "../../components/charts/LineChart";
import { STATISTIC_HEIGHT } from "../../config";
import { createUseStyles } from "react-jss";

const useStyles = createUseStyles({
  statisticWrapper: {},
});

export default function AcousticStatistics({
  imageStatistics,
  graphStatistics,
  audioStatistics,
}: {
  imageStatistics: ImageStatisticInterface[];
  graphStatistics: GraphStatisticInterface[];
  audioStatistics: AudioStatisticInterface[];
}): ReactElement {
  const classes = useStyles();
  const learningRates = graphStatistics.filter((graphStatistic) => {
    return graphStatistic.name === "lr";
  });
  const reconstructionStats = getCategoricalGraphStat(
    graphStatistics,
    "train_reconstruction_loss",
    "val_reconstruction_loss"
  );
  const melStats = getCategoricalGraphStat(
    graphStatistics,
    "train_mel_loss",
    "val_mel_loss"
  );
  const pitchStats = getCategoricalGraphStat(
    graphStatistics,
    "train_pitch_loss",
    "val_pitch_loss"
  );
  const prosodyStats = getCategoricalGraphStat(
    graphStatistics,
    "train_p_prosody_loss",
    "val_p_prosody_loss"
  );
  const durationStats = getCategoricalGraphStat(
    graphStatistics,
    "train_duration_loss",
    "val_duration_loss"
  );
  const ssimStats = getCategoricalGraphStat(
    graphStatistics,
    "train_ssim_loss",
    "val_ssim_loss"
  );
  const onlyTrainSpeakerEmb = graphStatistics.filter((graphStatistics) => {
    return graphStatistics.name === "only_train_speaker_emb";
  });
  const spectralDistortion = graphStatistics.filter((graphStatistic) => {
    return graphStatistic.name === "val_mcd_dtw";
  });
  const audiosSynthesized = audioStatistics.filter((audioStatistic) => {
    return audioStatistic.name === "val_wav_synthesized";
  });

  const melSpecsSynth = imageStatistics.filter((imageStatistic) => {
    return imageStatistic.name === "mel_spec_synth";
  });
  const melSpecsGT = imageStatistics.filter((imageStatistic) => {
    return imageStatistic.name === "mel_spec_ground_truth";
  });

  return (
    <>
      <Card title="Media" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12}>
            <ImageStatistic
              name="Ground Truth Mel-Spectrograms"
              steps={melSpecsGT.map((el) => {
                return el.step;
              })}
              paths={melSpecsGT.map((el) => {
                return el.path;
              })}
            ></ImageStatistic>
          </Col>
          <Col span={12}>
            <ImageStatistic
              name="Synthesized Mel-Spectrograms"
              steps={melSpecsSynth.map((el) => {
                return el.step;
              })}
              paths={melSpecsSynth.map((el) => {
                return el.path;
              })}
            ></ImageStatistic>
          </Col>
          <Col span={12}>
            <AudioStatistic
              name="Synthesized Audio"
              steps={audiosSynthesized.map((el) => {
                return el.step;
              })}
              paths={audiosSynthesized.map((el) => {
                return el.path;
              })}
            ></AudioStatistic>
          </Col>
        </Row>
      </Card>
      <Card title="Hyperparameters" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Learning Rate"
              steps={learningRates.map((el) => {
                return el.step;
              })}
              data={[
                learningRates.map((el) => {
                  return el.value;
                }),
              ]}
              labels={["Learning Rate"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Only Training Speaker Embeds"
              steps={onlyTrainSpeakerEmb.map((el) => {
                return el.step;
              })}
              data={[
                onlyTrainSpeakerEmb.map((el) => {
                  return el.value;
                }),
              ]}
              labels={["Only Training Speaker Embeds"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
        </Row>
      </Card>

      <Card title="Metrics" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Mel Spectral Distortion"
              steps={spectralDistortion.map((el) => {
                return el.step;
              })}
              data={[
                spectralDistortion.map((el) => {
                  return el.value;
                }),
              ]}
              labels={["Mel Spectral Distortion"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
        </Row>
      </Card>

      <Card title="Losses">
        <Row gutter={[20, 20]}>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Reconstruction Loss"
              steps={reconstructionStats["steps"]}
              data={reconstructionStats["data"]}
              labels={reconstructionStats["labels"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Mel Loss"
              steps={melStats["steps"]}
              data={melStats["data"]}
              labels={melStats["labels"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Structural Similarity Loss"
              steps={ssimStats["steps"]}
              data={ssimStats["data"]}
              labels={ssimStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Pitch Loss"
              steps={pitchStats["steps"]}
              data={pitchStats["data"]}
              labels={pitchStats["labels"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Prosody Loss"
              steps={prosodyStats["steps"]}
              data={prosodyStats["data"]}
              labels={prosodyStats["labels"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Duration Loss"
              steps={durationStats["steps"]}
              data={durationStats["data"]}
              labels={durationStats["labels"]}
              chartHeight={"100%"}
              chartWidth={"100%"}
              disableAnimation
            />
          </Col>
        </Row>
      </Card>
    </>
  );
}
