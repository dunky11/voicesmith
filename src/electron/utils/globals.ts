import { app } from "electron";
import path from "path";
import isDev from "electron-is-dev";
import { DB } from "./db";

export const UserDataPath = function () {
  let dataPath: string | null = null;
  return {
    getPath: function () {
      if (dataPath == null) {
        const path = DB.getInstance()
          .prepare("SELECT data_path AS dataPath FROM settings")
          .get().dataPath;
        if (path === null) {
          dataPath = app.getPath("userData");
        } else {
          dataPath = path;
        }
      }
      return dataPath;
    },
  };
};

const joinUserData = (pathToJoin: string) => () => {
  const userDataPath = UserDataPath().getPath();
  return path.join(userDataPath, pathToJoin);
};

export const BASE_PATH = app.getAppPath();
export const PORT = 12118;
export const getUserdataPath = joinUserData("");
export const getModelsDir = joinUserData("models");
export const getTrainingRunsDir = joinUserData("training_runs");
export const getAudioSynthDir = joinUserData("audio_synth");
export const getDatasetsDir = joinUserData("datasets");
export const getCleaningRunsDir = joinUserData("cleaning_runs");
export const getTextNormalizationRunsDir = joinUserData(
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
export const DB_PATH = path.join(
  app.getPath("userData"),
  "db",
  "voice_smith.db"
);
