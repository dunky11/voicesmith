import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import { startRun } from "../utils/processes";
import { exists } from "../utils/files";
import { ConfigurationInterface, SpeakerInterface } from "../../interfaces";
import {
  ASSETS_PATH,
  getDatasetsDir,
  DB_PATH,
  getModelsDir,
  getTrainingRunsDir,
  UserDataPath,
} from "../utils/globals";
import { DB, bool2int, getSpeakersWithSamples } from "../utils/db";
import { trainingRunInitialValues } from "../../config";

ipcMain.handle(
  "create-training-run",
  (event: IpcMainInvokeEvent, name: string) => {
    const info = DB.getInstance()
      .prepare(
        `INSERT INTO training_run 
      (
        name,  
        validation_size, 
        min_seconds,  
        max_seconds, 
        use_audio_normalization,
        acoustic_learning_rate, 
        acoustic_training_iterations, 
        acoustic_batch_size,
        acoustic_grad_accum_steps,
        acoustic_validate_every,
        vocoder_learning_rate,
        vocoder_training_iterations,
        vocoder_batch_size,
        vocoder_grad_accum_steps,
        vocoder_validate_every,
        only_train_speaker_emb_until,
        dataset_id,
        device
      ) VALUES(
        @name, 
        @validationSize, 
        @minSeconds, 
        @maxSeconds, 
        @useAudioNormalization,
        @acousticLearningRate, 
        @acousticTrainingIterations, 
        @acousticBatchSize, 
        @acousticGradAccumSteps,
        @acousticValidateEvery,
        @vocoderLearningRate, 
        @vocoderTrainingIterations, 
        @vocoderBatchSize, 
        @vocoderGradAccumSteps,
        @vocoderValidateEvery,
        @onlyTrainSpeakerEmbUntil,
        @datasetID,
        @device
      )`
      )
      .run(bool2int({ ...trainingRunInitialValues, name }));
    return info.lastInsertRowid;
  }
);

ipcMain.handle(
  "update-training-run-config",
  async (
    event: IpcMainInvokeEvent,
    config: ConfigurationInterface,
    ID: number
  ) => {
    DB.getInstance()
      .prepare(
        `UPDATE training_run SET
      name=@name, 
      validation_size=@validationSize, 
      min_seconds=@minSeconds, 
      max_seconds=@maxSeconds, 
      acoustic_learning_rate=@acousticLearningRate, 
      acoustic_training_iterations=@acousticTrainingIterations, 
      acoustic_batch_size=@acousticBatchSize,
      acoustic_grad_accum_steps=@acousticGradAccumSteps,
      acoustic_validate_every=@acousticValidateEvery,
      vocoder_learning_rate=@vocoderLearningRate,
      vocoder_training_iterations=@vocoderTrainingIterations,
      vocoder_batch_size=@vocoderBatchSize,
      vocoder_grad_accum_steps=@vocoderGradAccumSteps,
      vocoder_validate_every=@vocoderValidateEvery,
      only_train_speaker_emb_until=@onlyTrainSpeakerEmbUntil,
      dataset_id=@datasetID,
      device=@device
      WHERE ID=@trainingRunID`
      )
      .run(
        bool2int({
          ...config,
          trainingRunID: ID,
        })
      );
  }
);

ipcMain.handle(
  "remove-training-run",
  async (event: IpcMainInvokeEvent, ID: number) => {
    DB.getInstance().transaction(() => {
      DB.getInstance()
        .prepare("DELETE FROM audio_statistic WHERE training_run_id=@ID")
        .run({
          ID,
        });
      DB.getInstance()
        .prepare("DELETE FROM graph_statistic WHERE training_run_id=@ID")
        .run({
          ID,
        });
      DB.getInstance()
        .prepare("DELETE FROM image_statistic WHERE training_run_id=@ID")
        .run({
          ID,
        });
      DB.getInstance()
        .prepare("DELETE FROM training_run WHERE ID=@ID")
        .run({ ID: ID });
    })();
    const dir = path.join(getTrainingRunsDir(), String(ID));
    if (await exists(dir)) {
      await fsPromises.rmdir(dir, { recursive: true });
    }
  }
);

ipcMain.on("continue-training-run", (event: IpcMainEvent, runID: number) => {
  const datasetID = DB.getInstance()
    .prepare("SELECT dataset_id AS datasetID FROM training_run WHERE ID=@runID")
    .get({ runID }).datasetID;
  const speakers = getSpeakersWithSamples(datasetID);
  let sampleCount = 0;
  speakers.forEach((speaker: SpeakerInterface) => {
    sampleCount += speaker.samples.length;
  });
  if (sampleCount == 0) {
    event.reply("continue-run-reply", {
      type: "notEnoughSamples",
    });
    return;
  }
  startRun(event, "training_run.py", [
    "--training_run_id",
    String(runID),
    "--training_runs_path",
    getTrainingRunsDir(),
    "--assets_path",
    ASSETS_PATH,
    "--db_path",
    DB_PATH,
    "--models_path",
    getModelsDir(),
    "--datasets_path",
    getDatasetsDir(),
    "--user_data_path",
    UserDataPath().getPath(),
  ]);
});

ipcMain.handle(
  "fetch-training-run-names",
  async (event: IpcMainInvokeEvent, trainingRunID: number) => {
    let names;
    if (trainingRunID == null) {
      names = DB.getInstance().prepare("SELECT name FROM training_run").all();
    } else {
      names = DB.getInstance()
        .prepare("SELECT name FROM training_run WHERE ID!=@trainingRunID")
        .all({ trainingRunID });
    }
    names = names.map((model: any) => model.name);
    return names;
  }
);

ipcMain.handle(
  "fetch-training-run-configuration",
  (event: IpcMainInvokeEvent, trainingRunID: number) => {
    const configuration = DB.getInstance()
      .prepare(
        `SELECT 
        name,
        validation_size AS validationSize,
        min_seconds AS minSeconds, 
        max_seconds AS maxSeconds,
        use_audio_normalization AS useAudioNormalization,
        acoustic_learning_rate AS acousticLearningRate,
        acoustic_training_iterations AS acousticTrainingIterations,
        acoustic_batch_size AS acousticBatchSize,
        acoustic_grad_accum_steps AS acousticGradAccumSteps,
        acoustic_validate_every AS acousticValidateEvery,
        vocoder_learning_rate AS vocoderLearningRate,
        vocoder_training_iterations AS vocoderTrainingIterations,
        vocoder_batch_size AS vocoderBatchSize,
        vocoder_grad_accum_steps AS vocoderGradAccumSteps,
        vocoder_validate_every AS vocoderValidateEvery,
        only_train_speaker_emb_until AS onlyTrainSpeakerEmbUntil,
        dataset_id AS datasetID,
        device
      FROM training_run WHERE ID=@trainingRunID`
      )
      .get({ trainingRunID });
    return configuration;
  }
);

ipcMain.handle(
  "fetch-training-run-statistics",
  (event: IpcMainInvokeEvent, trainingRunID: number, stage: string) => {
    const graphStatistics = DB.getInstance()
      .prepare(
        `SELECT name, step, value FROM graph_statistic WHERE training_run_id=@trainingRunID AND stage=@stage`
      )
      .all({
        trainingRunID,
        stage,
      });
    const imageStatistics = DB.getInstance()
      .prepare(
        `SELECT name, step FROM image_statistic WHERE training_run_id=@trainingRunID AND stage=@stage`
      )
      .all({
        trainingRunID,
        stage,
      })
      .map((el: any) => ({
        ...el,
        path: path.join(
          getTrainingRunsDir(),
          String(trainingRunID),
          "image_logs",
          el.name,
          `${el.step}.png`
        ),
      }));
    const audioStatistics = DB.getInstance()
      .prepare(
        `SELECT name, step FROM audio_statistic WHERE training_run_id=@trainingRunID AND stage=@stage`
      )
      .all({
        trainingRunID,
        stage,
      })
      .map((el: any) => ({
        ...el,
        path: path.join(
          getTrainingRunsDir(),
          String(trainingRunID),
          "audio_logs",
          el.name,
          `${el.step}.flac`
        ),
      }));
    return {
      graphStatistics,
      imageStatistics,
      audioStatistics,
    };
  }
);

ipcMain.handle("fetch-training-runs", async () => {
  const trainingRuns = DB.getInstance()
    .prepare(
      `SELECT training_run.ID AS ID, training_run.name AS name, stage, dataset.name AS datasetName FROM training_run LEFT JOIN dataset ON training_run.dataset_id = dataset.ID`
    )
    .all();
  return trainingRuns;
});

ipcMain.handle(
  "fetch-training-run-progress",
  (event: IpcMainInvokeEvent, trainingRunID: number) => {
    const progress = DB.getInstance()
      .prepare(
        `SELECT 
        stage, 
        preprocessing_stage AS preprocessingStage, 
        preprocessing_copying_files_progress AS preprocessingCopyingFilesProgress,
        preprocessing_gen_vocab_progress AS preprocessingGenVocabProgress,
        preprocessing_gen_align_progress AS preprocessingGenAlignProgress,
        preprocessing_extract_data_progress AS preprocessingExtractDataProgress,
        acoustic_fine_tuning_progress AS acousticFineTuningProgress,
        ground_truth_alignment_progress AS groundTruthAlignmentProgress,
        vocoder_fine_tuning_progress AS vocoderFineTuningProgress 
      FROM training_run WHERE ID=@trainingRunID`
      )
      .get({ trainingRunID });
    return progress;
  }
);
