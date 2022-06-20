import { TrainingRunInterface } from "./interfaces";

export const INSTALL_BACKEND_CHANNEL = {
  IN: "install-backend",
  REPLY: "install-backend-reply",
};

export const START_SERVER_CHANNEL = {
  IN: "start-server",
};

export const FETCH_HAS_DOCKER_CHANNEL = {
  IN: "fetch-has-docker",
};

export const FETCH_NEEDS_INSTALL_CHANNEL = {
  IN: "fetch-needs-install",
};

export const FINISH_INSTALL_CHANNEL = {
  IN: "finish-install",
};

export const CONTINUE_CLEANING_RUN_CHANNEL = {
  IN: "continue-cleaning-run",
  REPLY: "run-reply",
};

export const FETCH_CLEANING_RUN_CHANNEL = {
  IN: "fetch-cleaning-run",
};

export const FETCH_CLEANING_RUN_CONFIG_CHANNEL = {
  IN: "fetch-cleaning-run-config",
};

export const UPDATE_CLEANING_RUN_CONFIG_CHANNEL = {
  IN: "update-cleaning-run-config",
};

export const FETCH_NOISY_SAMPLES_CHANNEL = {
  IN: "fetch-noisy-samples",
};

export const REMOVE_NOISY_SAMPLES_CHANNEL = {
  IN: "remove-noisy-samples",
};

export const FINISH_CLEANING_RUN_CHANNEL = {
  IN: "finish-cleaning-run",
  REPLY: "finish-cleaning-run-reply",
};

export const FETCH_DATASET_CANDIATES_CHANNEL = {
  IN: "fetch-dataset-candidates",
};

export const REMOVE_TRAINING_RUN_CHANNEL = {
  IN: "remove-training-run",
};

export const CONTINUE_TRAINING_RUN_CHANNEL = {
  IN: "continue-training-run",
  REPLY: "run-reply",
};

export const FETCH_TRAINING_RUN_NAMES_CHANNEL = {
  IN: "fetch-training-run-names",
};

export const FETCH_TRAINING_RUNS_CHANNEL = {
  IN: "fetch-training-runs",
};

export interface FETCH_TRAINING_RUNS_CHANNEL_TYPES {
  IN: {
    ARGS: {
      withStatistics: boolean;
      ID: number | null;
    };
    OUT: TrainingRunInterface[];
  };
}

export const CREATE_TRAINING_RUN_CHANNEL = {
  IN: "create-training-run",
};

export const UPDATE_TRAINING_RUN_CHANNEL = {
  IN: "update-training-run-config",
};

export const FETCH_DATASETS_CHANNEL = {
  IN: "fetch-datasets",
};

export const ADD_SPEAKER_CHANNEL = {
  IN: "add-speaker",
};

export const CREATE_DATASET_CHANNEL = {
  IN: "create-dataset",
};

export const REMOVE_DATASET_CHANNEL = {
  IN: "remove-dataset",
};

export const EXPORT_DATASET_CHANNEL = {
  IN: "export-dataset",
  REPLY: "export-dataset-reply",
  PROGRESS_REPLY: "export-dataset-progress-reply",
};

export const EDIT_DATASET_NAME_CHANNEL = {
  IN: "export-dataset-name",
};

export const ADD_SAMPLES_CHANNEL = {
  IN: "add-samples",
};

export const REMOVE_SAMPLES_CHANNEL = {
  IN: "remove-samples",
};

export const EDIT_SPEAKER_CHANNEL = {
  IN: "edit-speaker",
};

export const PICK_SPEAKERS_CHANNEL = {
  IN: "pick-speakers",
  REPLY: "pick-speakers-reply",
  PROGRESS_REPLY: "pick-speakers-progress-reply",
};

export const REMOVE_SPEAKERS_CHANNEL = {
  IN: "remove-speakers",
};

export const PICK_SPEAKER_FILES_CHANNEL = {
  IN: "pick-speaker-files",
};

export const FETCH_DATASET_CANDIDATES_CHANNEL = {
  IN: "fetch-dataset-candidates",
};

export const EDIT_SAMPLE_TEXT_CHANNEL = {
  IN: "edit-sample-text",
};

export const FETCH_DATASET_CHANNEL = {
  IN: "fetch-dataset",
};

export const FETCH_MODELS_CHANNEL = {
  IN: "fetch-models",
};

export const REMOVE_MODEL_CHANNEL = {
  IN: "remove-model",
};

export const CONTINUE_TEXT_NORMALIZATION_RUN_CHANNEL = {
  IN: "continue-text-normalization-sample",
  REPLY: "run-reply",
};

export const EDIT_TEXT_NORMALIZATION_SAMPLE_NEW_TEXT_CHANNEL = {
  IN: "edit-text-normalization-sample-new-text",
};

export const FETCH_TEXT_NORMALIZATION_RUN_CHANNEL = {
  IN: "fetch-text-normalization-run",
};

export const UPDATE_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL = {
  IN: "update-text-normalization-run-config",
};

export const FETCH_TEXT_NORMALIZATION_RUN_CONFIG_CHANNEL = {
  IN: "fetch-text-normalization-run-config",
};

export const FETCH_TEXT_NORMALIZATION_SAMPLES_CHANNEL = {
  IN: "fetch-text-normalization-samples",
};

export const REMOVE_TEXT_NORMALIZATION_SAMPLES_CHANNEL = {
  IN: "remove-text-normalization-samples",
};

export const GET_IMAGE_DATA_URL_CHANNEL = {
  IN: "get-image-data-url",
};

export const GET_AUDIO_DATA_URL_CHANNEL = {
  IN: "get-audio-data-url",
};

export const FETCH_LOGFILE_CHANNEL = {
  IN: "fetch-logfile",
};

export const EXPORT_FILES_CHANNEL = {
  IN: "export-files",
};

export const PICK_SINGLE_FOLDER_CHANNEL = {
  IN: "pick-single-folder",
};

export const EXPORT_FILE_CHANNEL = {
  IN: "export-file",
};

export const CREATE_PREPROCESSING_RUN_CHANNEL = {
  IN: "create-preprocessing-run",
};

export const FETCH_PREPROCESSING_RUNS_CHANNEL = {
  IN: "fetch-preprocessing-runs",
};

export const EDIT_PREPROCESSING_RUN_NAME_CHANNEL = {
  IN: "edit-preprocessing-run-name",
};

export const REMOVE_PREPROCESSING_RUN_CHANNEL = {
  IN: "remove-preprocessing-run",
};

export const FETCH_PREPROCESSING_NAMES_USED_CHANNEL = {
  IN: "fetch-preprocessing-names-used",
};

export const GET_APP_INFO_CHANNEL = {
  IN: "get-app-info",
};

export const SAVE_SETTINGS_CHANNEL = {
  IN: "save-settings",
  REPLY: "save-settings-reply",
};

export const FETCH_SETTINGS_CHANNEL = {
  IN: "fetch-settings",
};

export const FETCH_AUDIOS_SYNTH_CHANNEL = {
  IN: "fetch-audios-synth",
};

export const REMOVE_AUDIOS_SYNTH_CHANNEL = {
  IN: "remove-audios-synth",
};

export const FINISH_TEXT_NORMALIZATION_RUN_CHANNEL = {
  IN: "finish-text-normalization-run",
};

export const STOP_RUN_CHANNEL = {
  IN: "stop-run",
};

export const CONTINUE_SAMPLE_SPLITTING_RUN_CHANNEL = {
  IN: "continue-sample-splitting-run",
};

export const UPDATE_SAMPLE_SPLITTING_SAMPLE_CHANNEL = {
  IN: "update-sample-splitting-sample",
};

export const FETCH_SAMPLE_SPLITTING_SAMPLES_CHANNEL = {
  IN: "fetch-sample-splitting-samples",
};

export const REMOVE_SAMPLE_SPLITTING_SAMPLES_CHANNEL = {
  IN: "remove-sample-splitting-sample",
};

export const UPDATE_SAMPLE_SPLITTING_RUN_STAGE_CHANNEL = {
  IN: "update-sample-splitting-run-stage",
};

export const FETCH_SAMPLE_SPLITTING_RUNS_CHANNEL = {
  IN: "fetch-sample-splitting-runs",
};

export const UPDATE_SAMPLE_SPLITTING_RUN_CHANNEL = {
  IN: "update-sample-splitting-run",
};

export const REMOVE_SAMPLE_SPLITTING_SPLITS_CHANNEL = {
  IN: "remove-sample-splitting-splits",
};

export const START_BACKEND_CHANNEL = {
  IN: "start-backend-channel",
};
