import { app } from "electron";
import path from "path";
import isDev from "electron-is-dev";

export const BASE_PATH = app.getAppPath();
export const PORT = 12118;
export const USER_DATA_PATH = app.getPath("userData");
export const MODELS_DIR = path.join(USER_DATA_PATH, "models");
export const TRAINING_RUNS_DIR = path.join(USER_DATA_PATH, "training_runs");
export const AUDIO_SYNTH_DIR = path.join(USER_DATA_PATH, "audio_synth");
export const DATASET_DIR = path.join(USER_DATA_PATH, "datasets");
export const PREPROCESSING_RUNS_DIR = path.join(
  USER_DATA_PATH,
  "preprocessing_runs"
);
export const CLEANING_RUNS_DIR = path.join(
  PREPROCESSING_RUNS_DIR,
  "cleaning_runs"
);
export const TEXT_NORMALIZATION_RUNS_DIR = path.join(
  PREPROCESSING_RUNS_DIR,
  "text_normalization_runs"
);
export const PY_DIST_FOLDER = "backend_dist";
export const PY_FOLDER = "voice_smith";
export const INSTALLED_PATH = path.join(
  isDev ? BASE_PATH : process.resourcesPath,
  "INSTALLED"
);
export const POETRY_PATH = path.join(
  isDev ? BASE_PATH : process.resourcesPath,
  "backend"
);
export const ASSETS_PATH = path.join(
  isDev ? BASE_PATH : process.resourcesPath,
  "assets"
);
export const BACKEND_PATH = path.join(POETRY_PATH, "voice_smith");
export const DB_PATH = path.join(USER_DATA_PATH, "voice_smith.db");
