import React, { ReactElement } from "react";
import { Card, Row, Col } from "antd";
import {
  AudioStatisticInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
} from "../../interfaces";
import AudioStatistic from "./AudioStatistic";
import ImageStatistic from "./ImageStatistic";
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
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "lr";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Learning Rate"]}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Only Train Speaker Embeddings"
              lines={[
                graphStatistics.filter((graphStatistics) => {
                  return graphStatistics.name === "only_train_speaker_emb";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Speaker Embeds"]}
            />
          </Col>
        </Row>
      </Card>
      <Card title="Metrics" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Mel Spectral Distortion"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_mcd_dtw";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Mel Spectral Distortion"]}
            />
          </Col>
        </Row>
      </Card>
      <Card title="Losses">
        <Row gutter={[20, 20]}>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Reconstruction Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_reconstruction_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_reconstruction_loss";
                }),
              ]}
              labels={["Training Loss", "Validation Loss"]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Mel Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_mel_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_mel_loss";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Structural Similarity Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_ssim_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_ssim_loss";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Pitch Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_pitch_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_pitch_loss";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Prosody Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_p_prosody_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_p_prosody_loss";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12} className={classes.statisticWrapper}>
            <LineChart
              title="Duration Loss"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_duration_loss";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_duration_loss";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
        </Row>
      </Card>
    </>
  );
}
