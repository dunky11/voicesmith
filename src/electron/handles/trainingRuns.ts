import { ipcMain, IpcMainInvokeEvent, IpcMainEvent } from "electron";
import path from "path";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import { startRun } from "../utils/processes";
import { exists } from "../utils/files";
import {
  SpeakerInterface,
  ContinueTrainingRunReplyInterface,
  TrainingRunInterface,
  GraphStatisticInterface,
  ImageStatisticInterface,
  AudioStatisticInterface,
} from "../../interfaces";
import {
  REMOVE_TRAINING_RUN_CHANNEL,
  CONTINUE_TRAINING_RUN_CHANNEL,
  FETCH_TRAINING_RUN_NAMES_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL,
  CREATE_TRAINING_RUN_CHANNEL,
  UPDATE_TRAINING_RUN_CHANNEL,
  FETCH_TRAINING_RUNS_CHANNEL_TYPES,
} from "../../channels";
import { getTrainingRunsDir } from "../utils/globals";
import { DB, bool2int, getSpeakersWithSamples } from "../utils/db";
import { trainingRunInitialValues } from "../../config";

ipcMain.handle(
  CREATE_TRAINING_RUN_CHANNEL.IN,
  (event: IpcMainInvokeEvent, name: string) => {
    const info = DB.getInstance()
      .prepare(
        `INSERT INTO training_run 
      (
        name,
        maximum_workers,
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
        device,
        skip_on_error,
        acoustic_model_type
      ) VALUES(
        @name,
        @maximumWorkers,
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
        @device,
        @skipOnError,
        @acousticModelType
      )`
      )
      .run(bool2int({ ...trainingRunInitialValues, name }));
    return info.lastInsertRowid;
  }
);

ipcMain.handle(
  UPDATE_TRAINING_RUN_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, trainingRun: TrainingRunInterface) => {
    const config = { ...trainingRun.configuration };
    const temp = { ...trainingRun };
    delete temp.configuration;
    const flattened = {
      ...temp,
      ...config,
    };
    DB.getInstance()
      .prepare(
        `UPDATE training_run SET
      name=@name,
      maximum_workers=@maximumWorkers,
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
      forced_alignment_batch_size=@forcedAlignmentBatchSize,
      dataset_id=@datasetID,
      device=@device,
      skip_on_error=@skipOnError,
      acoustic_model_type=@acousticModelType
      WHERE ID=@ID`
      )
      .run(bool2int(flattened));
  }
);

ipcMain.handle(
  FETCH_TRAINING_RUNS_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    { stage, ID }: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["ARGS"]
  ): Promise<FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"]> => {
    const selectTrainingRunsStmt = DB.getInstance().prepare(`
      SELECT
      training_run.ID AS ID,
      training_run.name AS name,
      maximum_workers AS maximumWorkers,
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
      stage,
      preprocessing_stage AS preprocessingStage,
      preprocessing_copying_files_progress AS preprocessingCopyingFilesProgress,
      preprocessing_gen_vocab_progress AS preprocessingGenVocabProgress,
      preprocessing_gen_align_progress AS preprocessingGenAlignProgress,
      preprocessing_extract_data_progress AS preprocessingExtractDataProgress,
      acoustic_fine_tuning_progress AS acousticFineTuningProgress,
      ground_truth_alignment_progress AS groundTruthAlignmentProgress,
      vocoder_fine_tuning_progress AS vocoderFineTuningProgress,
      dataset_id AS datasetID,
      dataset.name AS datasetName,
      device,
      skip_on_error AS skipOnError,
      forced_alignment_batch_size AS forcedAlignmentBatchSize,
      acoustic_model_type AS acousticModelType
    FROM training_run
    LEFT JOIN dataset ON training_run.dataset_id = dataset.ID 
    ${ID === null ? "" : "WHERE training_run.ID=@ID"}`);
    const selectGraphStatisticsStmt = DB.getInstance().prepare(
      `SELECT name, step, value FROM graph_statistic WHERE training_run_id=@ID AND stage=@stage`
    );
    const selectImageStatisticsStmt = DB.getInstance().prepare(
      `SELECT name, step FROM image_statistic WHERE training_run_id=@ID AND stage=@stage`
    );
    const selectAudioStatisticsStmt = DB.getInstance().prepare(
      `SELECT name, step FROM audio_statistic WHERE training_run_id=@ID AND stage=@stage`
    );
    let runs;

    if (ID === null) {
      runs = selectTrainingRunsStmt.all();
    } else {
      runs = [selectTrainingRunsStmt.get({ ID })];
    }
    const trainingRuns: FETCH_TRAINING_RUNS_CHANNEL_TYPES["IN"]["OUT"] =
      runs.map((el: any) => {
        let graphStatistics: GraphStatisticInterface[] = [];
        let imageStatistics: ImageStatisticInterface[] = [];
        let audioStatistics: AudioStatisticInterface[] = [];
        if (stage !== null) {
          graphStatistics = selectGraphStatisticsStmt.all({
            ID: el.ID,
            stage,
          });
          imageStatistics = selectImageStatisticsStmt
            .all({
              ID: el.ID,
              stage,
            })
            .map((statsEl: any) => ({
              ...statsEl,
              path: path.join(
                getTrainingRunsDir(),
                String(el.ID),
                "image_logs",
                statsEl.name,
                `${statsEl.step}.png`
              ),
            }));
          audioStatistics = selectAudioStatisticsStmt
            .all({
              ID: el.ID,
              stage,
            })
            .map((statsEl: any) => ({
              ...statsEl,
              path: path.join(
                getTrainingRunsDir(),
                String(el.ID),
                "audio_logs",
                statsEl.name,
                `${statsEl.step}.flac`
              ),
            }));
        }

        const run: TrainingRunInterface = {
          ID: el.ID,
          type: "trainingRun",
          name: el.name,
          stage: el.stage,
          preprocessingStage: el.preprocessingStage,
          imageStatistics,
          audioStatistics,
          graphStatistics,
          preprocessingCopyingFilesProgress:
            el.preprocessingCopyingFilesProgress,
          preprocessingGenVocabProgress: el.preprocessingGenVocabProgress,
          preprocessingGenAlignProgress: el.preprocessingGenAlignProgress,
          preprocessingExtractDataProgress: el.preprocessingExtractDataProgress,
          acousticFineTuningProgress: el.acousticFineTuningProgress,
          groundTruthAlignmentProgress: el.groundTruthAlignmentProgress,
          vocoderFineTuningProgress: el.vocoderFineTuningProgress,
          configuration: {
            name: el.name,
            maximumWorkers: el.maximumWorkers,
            validationSize: el.validationSize,
            minSeconds: el.minSeconds,
            maxSeconds: el.maxSeconds,
            useAudioNormalization: el.useAudioNormalization,
            acousticLearningRate: el.acousticLearningRate,
            acousticTrainingIterations: el.acousticTrainingIterations,
            acousticBatchSize: el.acousticBatchSize,
            acousticGradAccumSteps: el.acousticGradAccumSteps,
            acousticValidateEvery: el.acousticValidateEvery,
            vocoderLearningRate: el.vocoderLearningRate,
            vocoderTrainingIterations: el.vocoderTrainingIterations,
            vocoderBatchSize: el.vocoderBatchSize,
            vocoderGradAccumSteps: el.vocoderGradAccumSteps,
            vocoderValidateEvery: el.vocoderValidateEvery,
            onlyTrainSpeakerEmbUntil: el.onlyTrainSpeakerEmbUntil,
            device: el.device,
            datasetID: el.datasetID,
            datasetName: el.datasetName,
            skipOnError: el.skipOnError === 1,
            forcedAlignmentBatchSize: el.forcedAlignmentBatchSize,
            acousticModelType: el.acousticModelType
          },
          canStart: el.datasetID !== null,
        };
        return run;
      });
    return trainingRuns;
  }
);

ipcMain.handle(
  REMOVE_TRAINING_RUN_CHANNEL.IN,
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

ipcMain.on(
  CONTINUE_TRAINING_RUN_CHANNEL.IN,
  (event: IpcMainEvent, runID: number) => {
    const datasetID = DB.getInstance()
      .prepare(
        "SELECT dataset_id AS datasetID FROM training_run WHERE ID=@runID"
      )
      .get({ runID }).datasetID;
    const speakers = getSpeakersWithSamples(datasetID, true);
    let sampleCount = 0;
    speakers.forEach((speaker: SpeakerInterface) => {
      sampleCount += speaker.samples.length;
    });
    if (sampleCount == 0) {
      const reply: ContinueTrainingRunReplyInterface = {
        type: "notEnoughSamples",
      };
      event.reply(CONTINUE_TRAINING_RUN_CHANNEL.REPLY, reply);
      return;
    }

    startRun(
      event,
      "./backend/voice_smith/training_run.py",
      ["--run_id", String(runID)],
      false
    );
  }
);

ipcMain.handle(
  FETCH_TRAINING_RUN_NAMES_CHANNEL.IN,
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
