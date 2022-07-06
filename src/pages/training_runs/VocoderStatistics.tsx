import React, { ReactElement } from "react";
import { Card, Row, Col } from "antd";
import {
  AudioStatisticInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
} from "../../interfaces";
import AudioStatistic from "./AudioStatistic";
import LineChart from "../../components/charts/LineChart";
import { STATISTIC_HEIGHT } from "../../config";

export default function VocoderStatistics({
  imageStatistics,
  graphStatistics,
  audioStatistics,
}: {
  imageStatistics: ImageStatisticInterface[];
  graphStatistics: GraphStatisticInterface[];
  audioStatistics: AudioStatisticInterface[];
}): ReactElement {
  const audiosReal = audioStatistics.filter((audioStatistic) => {
    return audioStatistic.name === "audio_real";
  });
  const audiosSynthesized = audioStatistics.filter((audioStatistic) => {
    return audioStatistic.name === "audio_fake";
  });
  return (
    <div>
      <Card title="Media" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12}>
            <AudioStatistic
              name="Real Audio"
              steps={audiosReal.map((el) => {
                return el.step;
              })}
              paths={audiosReal.map((el) => {
                return el.path;
              })}
            ></AudioStatistic>
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
          <Col span={12}>
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
        </Row>
      </Card>
      <Card title="Metrics" style={{ marginBottom: 16 }}>
        <Row gutter={[20, 20]}>
          <Col span={12}>
            <LineChart
              title="PESQ"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_pesq";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["PESQ"]}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="ESTOI"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_estoi";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["ESTOI"]}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="RMSE"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_rmse";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["RMSE"]}
            />
          </Col>
        </Row>
      </Card>
      <Card title="Losses">
        <Row gutter={[20, 20]}>
          <Col span={12}>
            <LineChart
              title="Total Loss Generator"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_total_loss_gen";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_total_loss_gen";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Total Loss Discriminator"
              lines={[
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "train_loss_disc_total";
                }),
                graphStatistics.filter((graphStatistic) => {
                  return graphStatistic.name === "val_loss_disc_total";
                }),
              ]}
              chartHeight={STATISTIC_HEIGHT}
              xLabel="Step"
              labels={["Training Loss", "Validation Loss"]}
            />
          </Col>
          <Col span={12}>
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
        </Row>
      </Card>
    </div>
  );
}
