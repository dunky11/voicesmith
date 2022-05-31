import React, { ReactElement } from "react";
import { Card, Row, Col } from "antd";
import {
  AudioStatisticInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
} from "../../interfaces";
import AudioStatistic from "./AudioStatistic";
import LineChart from "../../components/charts/LineChart";

export default function VocoderStatistics({
  imageStatistics,
  graphStatistics,
  audioStatistics,
}: {
  imageStatistics: ImageStatisticInterface[];
  graphStatistics: GraphStatisticInterface[];
  audioStatistics: AudioStatisticInterface[];
}): ReactElement {
  const learningRates = graphStatistics.filter((graphStatistic) => {
    return graphStatistic.name === "lr";
  });
  /** 
  const totalLossGenStats = getCategoricalGraphStat(
    graphStatistics,
    "train_total_loss_gen",
    "val_total_loss_gen"
  );
  const totalLossDiscStats = getCategoricalGraphStat(
    graphStatistics,
    "train_loss_disc_total",
    "val_loss_disc_total"
  );
  const melStats = getCategoricalGraphStat(
    graphStatistics,
    "train_mel_loss",
    "val_mel_loss"
  );
  const ssimStats = getCategoricalGraphStat(
    graphStatistics,
    "train_ssim_loss",
    "val_ssim_loss"
  );
  const featureMatchingStats = getCategoricalGraphStat(
    graphStatistics,
    "train_fm_loss",
    "val_fm_loss"
  );
  const genAdvMPDStats = getCategoricalGraphStat(
    graphStatistics,
    "train_gen_adv_loss_mpd",
    "val_gen_adv_loss_mpd"
  );
  const genAdvMSDStats = getCategoricalGraphStat(
    graphStatistics,
    "train_gen_adv_loss_msd",
    "val_gen_adv_loss_msd"
  );
  const discAdvMPDStats = getCategoricalGraphStat(
    graphStatistics,
    "train_disc_adv_loss_mpd",
    "val_disc_adv_loss_mpd"
  );
  const discAdvMSDStats = getCategoricalGraphStat(
    graphStatistics,
    "train_disc_adv_loss_msd",
    "val_disc_adv_loss_msd"
  );

  const audiosReal = audioStatistics.filter((audioStatistic) => {
    return audioStatistic.name === "audio_real";
  });
  const audiosSynthesized = audioStatistics.filter((audioStatistic) => {
    return audioStatistic.name === "audio_fake";
  });
  */
  return <></>;
  /**
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
              steps={learningRates.map((el) => {
                return el.step;
              })}
              data={[
                learningRates.map((el) => {
                  return el.value;
                }),
              ]}
              labels={["Learning Rate"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
        </Row>
      </Card>
      <Card title="Losses">
        <Row gutter={[20, 20]}>
          <Col span={12}>
            <LineChart
              title="Total Loss Generator"
              steps={totalLossGenStats["steps"]}
              data={totalLossGenStats["data"]}
              labels={totalLossGenStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Total Loss Discriminators"
              steps={totalLossDiscStats["steps"]}
              data={totalLossDiscStats["data"]}
              labels={totalLossDiscStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Mel Loss"
              steps={melStats["steps"]}
              data={melStats["data"]}
              labels={melStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Structural Similarity Loss"
              steps={ssimStats["steps"]}
              data={ssimStats["data"]}
              labels={ssimStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Generator Multi-Period-Discriminator Adverserial Loss"
              steps={genAdvMPDStats["steps"]}
              data={genAdvMPDStats["data"]}
              labels={genAdvMPDStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Generator Multi-Scale-Discriminator Adverserial Loss"
              steps={genAdvMSDStats["steps"]}
              data={genAdvMSDStats["data"]}
              labels={genAdvMSDStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Discriminators Multi-Period-Discriminator Adverserial Loss"
              steps={discAdvMPDStats["steps"]}
              data={discAdvMPDStats["data"]}
              labels={discAdvMPDStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Discriminators Multi-Scale-Discriminator Adverserial Loss"
              steps={discAdvMSDStats["steps"]}
              data={discAdvMSDStats["data"]}
              labels={discAdvMSDStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
          <Col span={12}>
            <LineChart
              title="Feature Matching Loss"
              steps={featureMatchingStats["steps"]}
              data={featureMatchingStats["data"]}
              labels={featureMatchingStats["labels"]}
              chartWidth={"100%"}
              chartHeight={"100%"}
            />
          </Col>
        </Row>
      </Card>
    </div>
  );
  */
}
