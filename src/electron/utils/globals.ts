import { app } from "electron";
import path from "path";
import isDev from "electron-is-dev";
import { DB } from "./db";

export const UserDataPath = function () {
  let dataPath: string | null = null;
  return {
    getPath: function () {
      if (dataPath === null) {
        const dataPathDB = DB.getInstance()
          .prepare("SELECT data_path AS dataPath FROM settings")
          .get().dataPath;
        if (dataPathDB === null) {
          dataPath = path.join(app.getPath("userData"), "data");
        } else {
          dataPath = dataPathDB;
        }
      }
      return dataPath;
    },
    setPath: function (path: string) {
      dataPath = path;
    },
  };
};

const joinUserData = (pathToJoin: string) => () => {
  const userDataPath = UserDataPath().getPath();
  return path.join(userDataPath, pathToJoin);
};

export const BASE_PATH = app.getAppPath();
export const PORT = 12118;
export const getModelsDir = joinUserData("models");
export const getTrainingRunsDir = joinUserData("training_runs");
export const getAudioSynthDir = joinUserData("audio_synth");
export const getDatasetsDir = joinUserData("datasets");
export const getCleaningRunsDir = joinUserData("cleaning_runs");
export const getTextNormalizationRunsDir = joinUserData(
  "text_normalization_runs"
);
export const getSampleSplittinRunsDir = joinUserData("sample_splitting_runs");
export const getInstalledPath = joinUserData("INSTALLED");
export const PY_DIST_FOLDER = "backend_dist";
export const PY_FOLDER = "voice_smith";
export const CONDA_PATH = path.join(
  isDev ? BASE_PATH : process.resourcesPath,
  "backend"
);
export const ASSETS_PATH = path.join(
  isDev ? BASE_PATH : process.resourcesPath,
  "assets"
);
export const BACKEND_PATH = path.join(CONDA_PATH, "voice_smith");
export const DB_PATH = path.join(
  app.getPath("userData"),
  "db",
  "voice_smith.db"
);
