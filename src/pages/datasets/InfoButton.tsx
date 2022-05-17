import React from "react";
import { Typography, Tree } from "antd";
import HelpButton from "../../components/help/HelpButton";

const treeData = [
  {
    title: "The Simpsons <- this is a dataset",
    key: "0",
    children: [
      {
        title: "Homer Simpson <- this is a speaker",
        key: "0-0",
        children: [
          {
            title:
              "dF4mNcv1.wav <- this is the audio part of the first sample from Homer Simpson",
            key: "0-0-0",
            isLeaf: true,
          },
          {
            title:
              "dF4mNcv1.txt  <- this is the text part of the first sample from Homer Simpson",
            key: "0-0-1",
            isLeaf: true,
          },
          { title: "M0J6zoHw.flac", key: "0-0-2", isLeaf: true },
          { title: "M0J6zoHw.txt", key: "0-0-3", isLeaf: true },
          { title: "8Sbff1iRz.flac", key: "0-0-4", isLeaf: true },
          { title: "8Sbff1iRz.txt", key: "0-0-5", isLeaf: true },
        ],
      },
      {
        title: "Bart Simpson <- this is a speaker",
        key: "0-1",
        children: [
          { title: "v1Jo6jhJ.flac", key: "0-1-0", isLeaf: true },
          { title: "v1Jo6jhJ.txt", key: "0-1-1", isLeaf: true },
          { title: "i2anU1mb.wav", key: "0-1-2", isLeaf: true },
          { title: "i2anU1mb.txt", key: "0-1-3", isLeaf: true },
          { title: "dS0exi6l.flac", key: "0-1-4", isLeaf: true },
          { title: "dS0exi6l.txt", key: "0-1-5", isLeaf: true },
        ],
      },
    ],
  },
];

export default function InfoButton({}: {}) {
  return (
    <HelpButton modalTitle="Help">
      <>
        <Typography.Title level={5}>
          How to structure the dataset?
        </Typography.Title>
        <Typography.Paragraph>
          Here you can add audio and text files to your dataset. The model will
          try to reproduce the audio from your dataset. VoiceSmith requires your
          dataset to follow a certain pattern. Each speaker is represented by a
          folder whose name is the speaker&apos;s name. The folder contains the
          samples of the speaker. Each sample consists of one audio and one text
          file. Both audio and text files have to have the same name. In the
          .txt files you simply write the transcription of the audio file. Below
          is an example of a correctly formatted dataset that contains two
          speakers with three samples each. As you can see supported audio
          extensions are .flac and .wav:
        </Typography.Paragraph>
        <Tree.DirectoryTree
          style={{ marginBottom: 24 }}
          defaultExpandAll
          treeData={treeData}
        />
        <Typography.Title level={5}>How to choose samples?</Typography.Title>
        <Typography.Paragraph>
          To achieve the best possible voice quality things you should consider
          are:
        </Typography.Paragraph>
        <ul>
          <li>
            All transcriptions have to be in english, which is currently the
            only supported language. This is due to ARPABET, an english only
            phone set that VoiceSmith uses.
          </li>
          <li>
            The text-to-speech model will simply try to recreate your dataset.
            If you want highly emotional speech you need highly emotional
            datasets. If you want your model to sound like text read from
            audiobooks you should use datasets scraped from audiobooks.
          </li>
          <li>
            More data is better. You can try with less but in order to create a
            realistic sounding voice you should have at least 300 samples
            (text/speech pairs).
          </li>
          <li>
            It may not be possible but try not to mix up too many different
            sources of audio for one speaker, as this will confuse the
            algorithm. For example, the sample person could sound very different
            when:
            <ul>
              <li>Giving an interview.</li>
              <li>Talking on the phone.</li>
              <li>Holding a speech.</li>
              <li>Reading text from a book.</li>
            </ul>
          </li>
          <li>
            For best quality make sure that your audio is sampled at a sampling
            rate of at least 22050hz, VoiceSmith will resample all audio but the
            resulting spectrograms won&apos;t contain higher frequency
            components which can lead to reduced audio quality (an example image
            is below). If you are not sure about the sampling rate of your audio
            you can right-click the files in your file explorer and inspect the
            properties.
          </li>
          <div
            style={{
              marginTop: 16,
              marginBottom: 16,
              display: "inline-block",
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <img
                src="/img/spec_no_high_freq.png"
                style={{
                  width: 300,
                  height: "auto",
                  marginBottom: 4,
                }}
              ></img>
              <Typography.Text style={{ fontSize: 12 }}>
                Spectrogram with cut off higher frequencies
              </Typography.Text>
            </div>
          </div>
          <li>
            Make sure your text is normalized. VoiceSmith will also normalize
            the text for you but it&apos;s better not to rely on it. Your text
            is normalized if everything is spelled out and contains only
            punctuation and alphabetic characters, some examples:
            <ul>
              <li>Mr. James goes to work. {"->"} Mister James goes to work.</li>
              <li>512$. {"->"} Five hundred twelve dollars.</li>
              <li>6:30PM. {"->"} Six thirty p m.</li>
            </ul>
          </li>
          <li>
            When creating a dataset try to use audio file formats that offer
            lossless compression like .flac or .wav. Other audio formats may
            reduce audio quality in exchange for a smaller file size.
          </li>
          <li>
            If your dataset contains audio files with a length of more than 10
            seconds, try splitting them up into chunks of a maximum of 10
            seconds each. On default, VoiceSmith will skip audio files that are
            longer then 10 seconds as this could cause out of memory issues (you
            can change this later in the training run configuration if you are
            sure whether you got enough RAM / VRAM).
          </li>
        </ul>
      </>
    </HelpButton>
  );
}
