export const DATASETS_ROUTE = {
  ROUTE: "/datasets",
  EDIT: {
    ROUTE: "/datasets/edit",
  },
  SELECTION: {
    ROUTE: "/datasets/dataset-selection",
  },
};

export const MODELS_ROUTE = {
  ROUTE: "/models",
  SELECTION: { ROUTE: "/models/selection" },
  SYNTHESIZE: { ROUTE: "/models/synthesize" },
};

export const TRAINING_RUNS_ROUTE = {
  ROUTE: "/training-runs",
  RUN_SELECTION: {
    ROUTE: "/training-runs/run-selection",
  },
  CREATE_MODEL: {
    ROUTE: "/training-runs/create-model",
    CONFIGURATION: {
      ROUTE: "/training-runs/create-model/configuration",
    },
    DATA_PREPROCESSING: {
      ROUTE: "/training-runs/create-model/data-preprocessing",
    },
    ACOUSTIC_TRAINING: {
      ROUTE: "/training-runs/create-model/acoustic-training",
    },
    GENERATE_GTA: {
      ROUTE: "/training-runs/create-model/generate-gta",
    },
    VOCODER_TRAINING: {
      ROUTE: "/training-runs/create-model/vocoder-training",
    },
    SAVE_MODEL: {
      ROUTE: "/training-runs/create-model/save-gta",
    },
  },
};

export const PREPROCESSING_RUNS_ROUTE = {
  ROUTE: "/preprocessing-runs",
  TEXT_NORMALIZATION: {
    ROUTE: "/preprocessing-runs/text-normalization",
    CONFIGURATION: {
      ROUTE: "/preprocessing-runs/text-normalization/configuration",
    },
    RUNNING: {
      ROUTE: "/preprocessing-runs/text-normalization/running",
    },
    CHOOSE_SAMPLES: {
      ROUTE: "/preprocessing-runs/text-normalization/choose-samples",
    },
  },
  DATASET_CLEANING: {
    ROUTE: "/preprocessing-runs/dataset-cleaning",
    CONFIGURATION: {
      ROUTE: "/preprocessing-runs/dataset-cleaning/configuration",
    },
    RUNNING: {
      ROUTE: "/preprocessing-runs/dataset-cleaning/running",
    },
    CHOOSE_SAMPLES: {
      ROUTE: "/preprocessing-runs/dataset-cleaning/choose-samples",
    },
    APPLY_CHANGES: {
      ROUTE: "/preprocessing-runs/dataset-cleaning/apply-changes",
    },
  },
  RUN_SELECTION: { ROUTE: "/preprocessing-runs/run-selection" },
  SAMPLE_SPLITTING: {
    ROUTE: "/preprocessing-runs/sample-splitting",
    CONFIGURATION: {
      ROUTE: "/preprocessing-runs/sample-splitting/configuration",
    },
    RUNNING: {
      ROUTE: "/preprocessing-runs/sample-splitting/running",
    },
    CHOOSE_SAMPLES: {
      ROUTE: "/preprocessing-runs/sample-splitting/choose-samples",
    },
    APPLY_CHANGES: {
      ROUTE: "/preprocessing-runs/sample-splitting/apply-changes",
    },
  },
};

export const SETTINGS_ROUTE = {
  ROUTE: "/settings",
};

export const RUN_QUEUE_ROUTE = {
  ROUTE: "/run-queue",
};

export const DOCUMENTATION_ROUTE = {
  INTODUCTION: { ROUTE: "/introduction" },
  DATASETS: { ROUTE: "/" },
};
