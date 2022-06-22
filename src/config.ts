import { TrainingRunConfigInterface, SpeakerInterface } from "./interfaces";

export const SERVER_URL = "http://localhost:12118";
export const POLL_LOGFILE_INTERVALL = 1000;
export const POLL_NULL_INTERVALL = 50;
export const CHART_BG_COLORS = [
  "rgb(255, 99, 132)",
  "rgb(54, 162, 235)",
  "rgb(255, 205, 86)",
];
export const CHART_BG_COLORS_FADED = [
  "rgba(255, 99, 132, 0.5)",
  "rgba(54, 162, 235, 0.5)",
  "rgba(255, 205, 86, 0.5)",
];
export const TEXT_EXTENSIONS = ["txt"];
export const AUDIO_EXTENSIONS = ["wav", "flac"];
export const STATISTIC_HEIGHT = 200;
export const DOCKER_IMAGE_NAME = "voicesmith/voicesmith:v0.2.1";
export const DOCKER_CONTAINER_NAME = "voice_smith";
export const CONDA_ENV_NAME = "voice_smith";
export const LANGUAGES: {
  name: string;
  iso6391: SpeakerInterface["language"];
}[] = [
  {
    name: "Bulgarian",
    iso6391: "bg",
  },
  {
    name: "Czech",
    iso6391: "cs",
  },
  {
    name: "German",
    iso6391: "de",
  },
  {
    name: "English",
    iso6391: "en",
  },
  {
    name: "Spanish",
    iso6391: "es",
  },
  {
    name: "French",
    iso6391: "fr",
  },
  {
    name: "Croatian",
    iso6391: "hr",
  },
  {
    name: "Polish",
    iso6391: "pl",
  },
  {
    name: "Portuguese",
    iso6391: "pt",
  },
  {
    name: "Russian",
    iso6391: "ru",
  },
  {
    name: "Swedish",
    iso6391: "sv",
  },
  {
    name: "Thai",
    iso6391: "th",
  },
  {
    name: "Turkish",
    iso6391: "tr",
  },
  {
    name: "Ukrainian",
    iso6391: "uk",
  },
];

export const trainingRunInitialValues: TrainingRunConfigInterface = {
  name: "",
  maximumWorkers: -1,
  validationSize: 5.0,
  minSeconds: 0.5,
  maxSeconds: 10,
  useAudioNormalization: true,
  acousticLearningRate: 0.0002,
  acousticTrainingIterations: 30000,
  acousticValidateEvery: 2000,
  acousticBatchSize: 5,
  acousticGradAccumSteps: 3,
  vocoderLearningRate: 0.0002,
  vocoderTrainingIterations: 20000,
  vocoderValidateEvery: 2000,
  vocoderBatchSize: 5,
  vocoderGradAccumSteps: 3,
  device: "CPU",
  onlyTrainSpeakerEmbUntil: 5000,
  datasetID: null,
  datasetName: null,
  skipOnError: false,
};

export const defaultPageOptions = {
  defaultPageSize: 100,
  pageSizeOptions: [50, 100, 250, 1000],
};
