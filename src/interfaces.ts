export interface WaveSurverInterface {
  load: (a: string) => void;
  pause: () => void;
  play: () => void;
  stop: () => void;
  on: (a: string, b: () => any) => void;
}

export interface StatisticInterface {
  name: string;
  entries: Array<
    ImageStatisticInterface | AudioStatisticInterface | GraphStatisticInterface
  >;
}

export interface ImageStatisticInterface {
  name: string;
  step: number;
  path: string;
}

export interface AudioStatisticInterface {
  name: string;
  step: number;
  path: string;
}

export interface GraphStatisticInterface {
  name: string;
  step: number;
  value: number;
}

export interface ModelSpeakerInterface {
  name: string;
  speakerID: number;
}

export interface ModelInterface {
  ID: number;
  name: string;
  type: "Delighful_FreGANv1_v0.0";
  createdAt: string;
  description: string;
  speakers: ModelSpeakerInterface[];
}

export interface DatasetInterface {
  ID: number;
  name: string;
  speakerCount?: number;
  speakers: SpeakerInterface[];
  referencedBy: string | null;
}

export interface SpeakerSampleInterface {
  ID?: number;
  txtPath: string;
  audioPath: string;
  text: string;
  fullAudioPath?: string;
}

export interface SpeakerInterface {
  ID: number;
  name: string;
  language:
    | "bg"
    | "cs"
    | "de"
    | "en"
    | "es"
    | "fr"
    | "hr"
    | "pl"
    | "pt"
    | "ru"
    | "sv"
    | "th"
    | "tr"
    | "uk";
  samples: SpeakerSampleInterface[];
}

export interface RunInterface {
  ID: number;
  name: string;
  type:
    | "trainingRun"
    | "textNormalizationRun"
    | "cleaningRun"
    | "sampleSplittingRun";
}

export interface SynthConfigInterface {
  text: string;
  speakerID: number | null;
  talkingSpeed: number;
}

export interface TrainingRunInterface extends RunInterface {
  ID: number;
  name: string;
  imageStatistics: ImageStatisticInterface[];
  audioStatistics: AudioStatisticInterface[];
  graphStatistics: GraphStatisticInterface[];
  stage:
    | "not_started"
    | "preprocessing"
    | "acoustic_fine_tuning"
    | "ground_truth_alignment"
    | "vocoder_fine_tuning"
    | "save_model"
    | "finished";
  configuration: TrainingRunConfigInterface;
  preprocessingStage:
    | "not_started"
    | "copying_files"
    | "gen_vocab"
    | "gen_alignments"
    | "extract_data"
    | "finished";
  preprocessingCopyingFilesProgress: number;
  preprocessingGenVocabProgress: number;
  preprocessingGenAlignProgress: number;
  preprocessingExtractDataProgress: number;
  acousticFineTuningProgress: number;
  groundTruthAlignmentProgress: number;
  vocoderFineTuningProgress: number;
  canStart: boolean;
}

export interface TrainingRunConfigInterface {
  name: string;
  maximumWorkers: number;
  validationSize: number;
  minSeconds: number;
  maxSeconds: number;
  useAudioNormalization: boolean;
  acousticLearningRate: number;
  acousticTrainingIterations: number;
  acousticBatchSize: number;
  acousticGradAccumSteps: number;
  acousticValidateEvery: number;
  vocoderLearningRate: number;
  vocoderTrainingIterations: number;
  vocoderBatchSize: number;
  vocoderGradAccumSteps: number;
  vocoderValidateEvery: number;
  device: "CPU" | "GPU";
  onlyTrainSpeakerEmbUntil: number;
  datasetID: number | null;
  datasetName: string | null;
  skipOnError: boolean;
}

export interface CleaningRunInterface extends RunInterface {
  ID: number;
  type: "cleaningRun";
  stage:
    | "not_started"
    | "copying_files"
    | "transcribe"
    | "choose_samples"
    | "apply_changes"
    | "finished";
  copyingFilesProgress: number;
  transcriptionProgress: number;
  applyingChangesProgress: number;
  configuration: CleaningRunConfigInterface;
  canStart: boolean;
}

export interface CleaningRunConfigInterface {
  name: string;
  datasetID?: number;
  datasetName: string;
  skipOnError: boolean;
  device: "CPU" | "GPU";
  maximumWorkers: -1;
}

export interface UsageStatsInterface {
  cpuUsage: number;
  totalRam: number;
  ramUsed: number;
  totalDisk: number;
  diskUsed: number;
}

export interface AudioSynthInterface {
  ID: number;
  filePath: string;
  text: string;
  speakerName: string;
  modelName: string;
  createdAt: string;
  samplingRate: number;
  durSecs: number;
}

export interface TextNormalizationRunInterface extends RunInterface {
  ID: number;
  type: "textNormalizationRun";
  stage: "not_started" | "text_normalization" | "choose_samples" | "finished";
  textNormalizationProgress: number;
  configuration: TextNormalizationRunConfigInterface;
  canStart: boolean;
}

export interface TextNormalizationRunConfigInterface {
  name: string;
  datasetID: number | null;
  datasetName: string | null;
}

export interface SampleSplittingRunInterface extends RunInterface {
  stage:
    | "not_started"
    | "copying_files"
    | "gen_vocab"
    | "gen_alignments"
    | "creating_splits"
    | "choose_samples"
    | "apply_changes"
    | "finished";
  copyingFilesProgress: number;
  genVocabProgress: number;
  genAlignProgress: number;
  creatingSplitsProgress: number;
  applyingChangesProgress: number;
  configuration: SampleSplittingRunConfigInterface;
  canStart: boolean;
}

export interface SampleSplittingRunConfigInterface {
  name: string;
  device: "CPU" | "GPU";
  datasetID: number | null;
  datasetName: string | null;
  skipOnError: boolean;
  maximumWorkers: number;
}

export interface FileInterface {
  path: string;
  extname: string;
  name: string;
  basename: string;
}

export interface NoisySampleInterface {
  ID: number;
  text: string;
  audioPath: string;
  labelQuality: number;
}

export interface TextNormalizationSampleInterface {
  ID: number;
  sampleID: number;
  oldText: string;
  newText: string;
  reason: string;
  audioPath: string;
}

export interface TerminalMessage {
  type: "message" | "error";
  message: string;
}

export interface SettingsInterface {
  dataPath: string | null;
}

export interface AppInfoInterface {
  platform:
    | "aix"
    | "darwin"
    | "freebsd"
    | "linux"
    | "openbsd"
    | "sunos"
    | "win32"
    | "android"
    | "haiku"
    | "cygwin"
    | "netbsd";
  version: string;
}

export interface InstallBackendReplyInterface {
  type: "message" | "error" | "finished";
  message: string;
  success?: boolean;
}

export interface ContinueTrainingRunReplyInterface {
  type: "notEnoughSpeakers" | "notEnoughSamples";
}

export interface FinishCleaningRunReplyInterface {
  type: "progress" | "finished";
  progress?: number;
}

export interface SampleSplittingSplitInterface {
  ID: number;
  text: string;
  audioPath: string;
}

export interface SampleSplittingSampleInterface {
  ID: number;
  speakerName: string;
  text: string;
  audioPath: string;
  splits: SampleSplittingSplitInterface[];
}

export interface InstallerOptionsInterface {
  device: "CPU" | "GPU";
  dockerIsInstalled: boolean | null;
  hasInstalledNCT: boolean;
}

export interface RunManagerInterface {
  isRunning: boolean;
  queue: RunInterface[];
}

export interface ImportSettingsInterface {
  language: SpeakerInterface["language"];
}

export interface NavigationSettingsInterface {
  isDisabled: boolean;
}

export type PreprocessingRunType =
  | TextNormalizationRunInterface
  | SampleSplittingRunInterface
  | CleaningRunInterface;
