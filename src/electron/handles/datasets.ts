import {
  ipcMain,
  IpcMainInvokeEvent,
  IpcMainEvent,
  OpenDialogOptions,
  dialog,
} from "electron";
import path from "path";
import fsNative from "fs";
const fsPromises = fsNative.promises;
import {
  safeUnlink,
  exists,
  safeMkdir,
  copyDir,
  safeRmDir,
} from "../utils/files";
import { getDatasetsDir } from "../utils/globals";
import { DB, getSpeakersWithSamples, getReferencedBy } from "../utils/db";
import {
  ADD_SPEAKER_CHANNEL,
  FETCH_DATASETS_CHANNEL,
  CREATE_DATASET_CHANNEL,
  REMOVE_DATASET_CHANNEL,
  EXPORT_DATASET_CHANNEL,
  EDIT_DATASET_NAME_CHANNEL,
  ADD_SAMPLES_CHANNEL,
  REMOVE_SAMPLES_CHANNEL,
  EDIT_SPEAKER_CHANNEL,
  PICK_SPEAKERS_CHANNEL,
  REMOVE_SPEAKERS_CHANNEL,
  PICK_SPEAKER_FILES_CHANNEL,
  FETCH_DATASET_CANDIDATES_CHANNEL,
  EDIT_SAMPLE_TEXT_CHANNEL,
  FETCH_DATASET_CHANNEL,
} from "../../channels";
import { AUDIO_EXTENSIONS, TEXT_EXTENSIONS } from "../../config";
import {
  SpeakerSampleInterface,
  FileInterface,
  DatasetInterface,
  SpeakerInterface,
  ImportSettingsInterface,
} from "../../interfaces";

ipcMain.handle(FETCH_DATASETS_CHANNEL.IN, async () => {
  const datasets = DB.getInstance()
    .prepare(
      `
        SELECT dataset.ID, dataset.name, count(speaker.dataset_id) AS speakerCount        
        FROM dataset
        LEFT JOIN speaker
        ON speaker.dataset_id = dataset.ID
        GROUP BY dataset.ID
      `
    )
    .all()
    .map((dataset: any) => ({
      ...dataset,
      referencedBy: getReferencedBy(dataset.ID),
    }));
  return datasets;
});

ipcMain.handle(
  ADD_SPEAKER_CHANNEL.IN,
  (event: IpcMainInvokeEvent, speakerName: string, datasetID: number) => {
    DB.getInstance()
      .prepare(
        `
      INSERT INTO speaker 
      (
        name, 
        dataset_id
      ) VALUES(@name, @datasetID)
      `
      )
      .run({
        name: speakerName,
        datasetID,
      });
  }
);

ipcMain.handle(
  CREATE_DATASET_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, name: string) => {
    DB.getInstance()
      .prepare(`INSERT INTO dataset (name) VALUES (@name)`)
      .run({ name });
  }
);

ipcMain.handle(
  REMOVE_DATASET_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, ID: number) => {
    const datasetPath = path.join(getDatasetsDir(), String(ID));
    await safeRmDir(datasetPath);
    const deleteDSstmt = DB.getInstance().prepare(
      `DELETE FROM dataset WHERE ID=@ID`
    );
    const deleteSpeakersStmt = DB.getInstance().prepare(
      `DELETE FROM speaker WHERE dataset_id=@ID`
    );
    const deleteSamplesStmt = DB.getInstance().prepare(
      `
      DELETE FROM sample
      WHERE sample.speaker_id IN
      (
          select speaker.ID
          from speaker
          where speaker.dataset_id=@ID
      )
      `
    );
    DB.getInstance().transaction(() => {
      deleteSamplesStmt.run({ ID });
      deleteSpeakersStmt.run({ ID });
      deleteDSstmt.run({ ID });
    })();
  }
);

ipcMain.on(
  EXPORT_DATASET_CHANNEL.IN,
  (event: IpcMainEvent, datasets: DatasetInterface[]) => {
    const options: OpenDialogOptions = {
      properties: ["openDirectory"],
    };

    dialog.showOpenDialog(null, options).then(async (response) => {
      if (response.canceled) {
        EXPORT_DATASET_CHANNEL.REPLY;
        return;
      }
      const outDatasetDir = response.filePaths[0];
      const getSpeakersStmt = DB.getInstance().prepare(
        `SELECT ID, name FROM speaker WHERE dataset_id=@datasetID`
      );
      const getTextsStmt = DB.getInstance().prepare(
        `SELECT text, txt_path AS txtPath FROM sample WHERE speaker_id=@speakerID`
      );
      const speakersToCopy: {
        inPath: string;
        outPath: string;
        speakerID: number;
      }[] = [];
      for (const dataset of datasets) {
        const speakers = getSpeakersStmt.all({ datasetID: dataset.ID });
        for (const speaker of speakers) {
          const inPath = path.join(
            getDatasetsDir(),
            String(dataset.ID),
            "speakers",
            String(speaker.ID)
          );
          const outPath = path.join(outDatasetDir, dataset.name, speaker.name);
          speakersToCopy.push({ inPath, outPath, speakerID: speaker.ID });
        }
      }
      const total = speakersToCopy.length;
      event.reply(EXPORT_DATASET_CHANNEL.PROGRESS_REPLY, 0, total);
      for (let i = 0; i < speakersToCopy.length; i++) {
        const speakerToCopy = speakersToCopy[i];
        await copyDir(speakerToCopy.inPath, speakerToCopy.outPath);
        const textFiles = getTextsStmt.all({
          speakerID: speakerToCopy.speakerID,
        });
        await Promise.all(
          textFiles.map((textFile: any) =>
            fsPromises.writeFile(
              path.join(speakerToCopy.outPath, textFile.txtPath),
              textFile.text
            )
          )
        );
        event.reply(EXPORT_DATASET_CHANNEL.PROGRESS_REPLY, i + 1, total);
      }
      event.reply(EXPORT_DATASET_CHANNEL.REPLY);
    });
  }
);

ipcMain.handle(
  EDIT_DATASET_NAME_CHANNEL.IN,
  (event: IpcMainInvokeEvent, ID: number, newName: string) => {
    DB.getInstance().prepare("UPDATE dataset SET name=@name WHERE ID=@ID").run({
      ID,
      name: newName,
    });
  }
);

const filesToSamples = async (audioFiles: string[], txtFiles: string[]) => {
  const audioExtensions = AUDIO_EXTENSIONS.map((ext) => `.${ext}`);
  const samples: SpeakerSampleInterface[] = [];
  const audioFilesSet: Set<string> = new Set(audioFiles);

  for (const txtFile of txtFiles) {
    const txtPathNoExt = txtFile.split(path.extname(txtFile))[0];
    for (const audioExtension of audioExtensions) {
      const audioFile = `${txtPathNoExt}${audioExtension}`;
      if (audioFilesSet.has(audioFile)) {
        const text = await fsPromises.readFile(txtFile, {
          encoding: "utf8",
        });
        samples.push({
          txtPath: txtFile,
          audioPath: audioFile,
          text: text,
        });
        audioFilesSet.delete(audioFile);
        break;
      }
    }
  }
  return samples;
};

ipcMain.handle(
  ADD_SAMPLES_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    speaker: SpeakerInterface,
    filePaths: FileInterface[],
    datasetID: number
  ) => {
    const currentSampleSet: Set<string> = new Set();

    speaker.samples.forEach((sample: SpeakerSampleInterface) => {
      currentSampleSet.add(sample.audioPath);
      currentSampleSet.add(sample.txtPath);
    });
    const filesCleaned = filePaths.filter((file: FileInterface) => {
      return !currentSampleSet.has(file.path);
    });

    const txtFiles: string[] = filesCleaned
      .filter((file: FileInterface) => {
        return file.extname === ".txt";
      })
      .map((file: FileInterface) => file.path);
    const audioFiles: string[] = filesCleaned
      .filter((file: FileInterface) => {
        return file.extname !== ".txt";
      })
      .map((file: FileInterface) => file.path);
    const stmt = DB.getInstance().prepare(
      "INSERT INTO sample (txt_path, audio_path, speaker_id, text) VALUES(@txtPath, @audioPath, @speakerId, @text)"
    );

    const samples = await filesToSamples(audioFiles, txtFiles);

    copySpeakerFiles(datasetID, speaker.ID, samples);

    const insertManySamples = DB.getInstance().transaction((els: any[]) => {
      for (const el of els) stmt.run(el);
    });

    insertManySamples(
      samples.map((sample) => ({
        txtPath: path.basename(sample.txtPath),
        audioPath: path.basename(sample.audioPath),
        speakerId: speaker.ID,
        text: sample.text,
      }))
    );
  }
);

ipcMain.handle(
  REMOVE_SAMPLES_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    datasetID: number,
    speakerID: number,
    samples: SpeakerSampleInterface[]
  ) => {
    const stmt = DB.getInstance().prepare("DELETE FROM sample WHERE ID=@ID");
    const deleteMany = DB.getInstance().transaction((els: any[]) => {
      for (const el of els) stmt.run(el);
    });

    deleteMany(
      samples.map((sample) => ({
        ID: sample.ID,
      }))
    );

    const speakerDir = path.join(
      getDatasetsDir(),
      String(datasetID),
      "speakers",
      String(speakerID)
    );

    for (const sample of samples) {
      await safeUnlink(path.join(speakerDir, sample.txtPath));
      await safeUnlink(path.join(speakerDir, sample.audioPath));
    }
  }
);

const copySpeakerFiles = async (
  datasetID: number,
  speakerID: number | bigint,
  samples: SpeakerSampleInterface[]
) => {
  const speakerDir = path.join(
    getDatasetsDir(),
    String(datasetID),
    "speakers",
    String(speakerID)
  );
  if (!(await exists(speakerDir))) {
    await safeMkdir(speakerDir);
  }
  for (const sample of samples) {
    await fsPromises.copyFile(
      sample.audioPath,
      path.join(speakerDir, path.basename(sample.audioPath))
    );
  }
};

ipcMain.on(
  PICK_SPEAKERS_CHANNEL.IN,
  async (
    event: IpcMainEvent,
    datasetID: number,
    importSettings: ImportSettingsInterface
  ) => {
    const options: OpenDialogOptions = {
      properties: ["openDirectory", "multiSelections"],
    };

    const textExtensions = TEXT_EXTENSIONS.map((ext) => `.${ext}`);
    const audioExtensions = AUDIO_EXTENSIONS.map((ext) => `.${ext}`);

    dialog.showOpenDialog(null, options).then(async (response) => {
      if (response.canceled) {
        event.reply(PICK_SPEAKERS_CHANNEL.REPLY);
        return;
      }
      const speakers = DB.getInstance()
        .prepare("SELECT name FROM speaker WHERE dataset_id=@datasetID")
        .all({ datasetID });
      const stmtSpeakerID = DB.getInstance().prepare(
        "SELECT ID FROM speaker WHERE dataset_id=@datasetID AND name=@name"
      );

      const stmtInsertSample = DB.getInstance().prepare(
        "INSERT OR IGNORE INTO sample (txt_path, audio_path, speaker_id, text) VALUES(@txtPath, @audioPath, @speakerId, @text)"
      );

      const stmtInsertSpeaker = DB.getInstance().prepare(
        "INSERT INTO speaker (name, dataset_id, language) VALUES (@name, @datasetID, @language)"
      );

      const insertManySamples = DB.getInstance().transaction((els: any[]) => {
        for (const el of els) stmtInsertSample.run(el);
      });

      const speakerNames = speakers.map((speaker: any) => speaker.name);
      for (
        let progress = 1;
        progress < response.filePaths.length + 1;
        progress++
      ) {
        const speakerPath = response.filePaths[progress - 1];
        const split = speakerPath.split(path.sep);
        const speakerName = split[split.length - 1];
        const textFiles: string[] = [];
        const audioFiles: string[] = [];
        const files = await fsPromises.readdir(speakerPath, {
          withFileTypes: true,
        });
        files.forEach((file) => {
          if (!file.isFile()) {
            return;
          }
          const filePath = path.join(speakerPath, file.name);
          const ext = path.extname(filePath);
          if (textExtensions.includes(ext)) {
            textFiles.push(filePath);
          } else if (audioExtensions.includes(ext)) {
            audioFiles.push(filePath);
          }
        });
        const samples = await filesToSamples(audioFiles, textFiles);

        if (samples.length === 0) {
          event.reply(
            "pick-speakers-progress-reply",
            progress,
            response.filePaths.length
          );
          continue;
        }
        if (speakerNames.includes(speakerName)) {
          const speakerID = stmtSpeakerID.get({
            datasetID,
            name: speakerName,
          }).ID;
          copySpeakerFiles(datasetID, speakerID, samples);
          insertManySamples(
            samples.map((sample) => ({
              txtPath: path.basename(sample.txtPath),
              audioPath: path.basename(sample.audioPath),
              speakerId: speakerID,
              text: sample.text,
            }))
          );
        } else {
          const info = stmtInsertSpeaker.run({
            name: speakerName,
            datasetID,
            language: importSettings.language,
          });
          const speakerID = info.lastInsertRowid;
          copySpeakerFiles(datasetID, speakerID, samples);
          insertManySamples(
            samples.map((sample) => ({
              txtPath: path.basename(sample.txtPath),
              audioPath: path.basename(sample.audioPath),
              speakerId: speakerID,
              text: sample.text,
            }))
          );
        }
        event.reply(
          PICK_SPEAKERS_CHANNEL.PROGRESS_REPLY,
          progress,
          response.filePaths.length
        );
      }
      event.reply(PICK_SPEAKERS_CHANNEL.REPLY);
    });
  }
);

ipcMain.handle(
  EDIT_SPEAKER_CHANNEL.IN,
  (event: IpcMainInvokeEvent, speaker: SpeakerInterface) => {
    DB.getInstance()
      .prepare("UPDATE speaker SET name=@name, language=@language WHERE ID=@ID")
      .run(speaker);
  }
);

ipcMain.handle(
  REMOVE_SPEAKERS_CHANNEL.IN,
  async (
    event: IpcMainInvokeEvent,
    datasetID: number,
    speakerIDs: number[]
  ) => {
    const stmtDeleteSpeaker = DB.getInstance().prepare(
      "DELETE FROM speaker WHERE ID=@speakerID"
    );
    const stmtDeleteSamples = DB.getInstance().prepare(
      "DELETE FROM sample WHERE speaker_id=@speakerID"
    );
    const deleteMany = DB.getInstance().transaction((IDs: number[]) => {
      for (const speakerID of IDs) {
        stmtDeleteSamples.run({ speakerID });
        stmtDeleteSpeaker.run({ speakerID });
      }
    });
    for (const speakerID of speakerIDs) {
      const speakerDir = path.join(
        getDatasetsDir(),
        String(datasetID),
        "speakers",
        String(speakerID)
      );
      if (await exists(speakerDir)) {
        safeRmDir(speakerDir);
      }
    }
    deleteMany(speakerIDs);
  }
);

ipcMain.handle(
  PICK_SPEAKER_FILES_CHANNEL.IN,
  async (event: IpcMainInvokeEvent) => {
    const files: FileInterface[] = await new Promise((resolve, reject) => {
      const options: OpenDialogOptions = {
        properties: ["openFile", "multiSelections"],
        filters: [
          {
            name: "all",
            extensions: [...TEXT_EXTENSIONS, ...AUDIO_EXTENSIONS],
          },
        ],
      };
      dialog.showOpenDialog(null, options).then((response) => {
        if (response.canceled) {
          resolve([]);
          return;
        }
        resolve(
          response.filePaths.map((filePath: string) => {
            const p = path.parse(filePath);
            return {
              path: filePath,
              extname: p.ext,
              name: p.name,
              basename: p.base,
            };
          })
        );
      });
    });
    return files;
  }
);

ipcMain.handle(FETCH_DATASET_CANDIDATES_CHANNEL.IN, () => {
  const datasets = DB.getInstance()
    .prepare(
      `SELECT  
          dataset.ID AS ID,
          dataset.name AS name,
          training_run.name AS trainingRunName,
          cleaning_run.name AS cleaningRunName,
          text_normalization_run.name AS textNormalizationName,
          sample_splitting_run.name AS sampleSplittingRunName
        FROM dataset 
        LEFT JOIN training_run ON training_run.dataset_id = dataset.ID
        LEFT JOIN cleaning_run ON cleaning_run.dataset_id = dataset.ID
        LEFT JOIN text_normalization_run ON text_normalization_run.dataset_id = dataset.ID
        LEFT JOIN sample_splitting_run ON sample_splitting_run.dataset_id = dataset.ID
        WHERE EXISTS (
          SELECT 1 FROM speaker WHERE speaker.dataset_id = dataset.ID AND EXISTS(
              SELECT 1 FROM sample WHERE sample.speaker_id = speaker.ID
          )
        )
      `
    )
    .all()
    .map((dataset: any) => {
      let referencedBy = null;
      if (dataset.trainingRunName !== null) {
        referencedBy = dataset.trainingRunName;
      } else if (dataset.cleaningRunName !== null) {
        referencedBy = dataset.cleaningRunName;
      } else if (dataset.textNormalizationName !== null) {
        referencedBy = dataset.textNormalizationName;
      } else if (dataset.sampleSplittingRunName) {
        referencedBy = dataset.sampleSplittingRunName;
      }
      return {
        ID: dataset.ID,
        name: dataset.name,
        speakers: [] as SpeakerInterface[],
        referencedBy,
      };
    });
  return datasets;
});

ipcMain.handle(
  EDIT_SAMPLE_TEXT_CHANNEL.IN,
  (event: IpcMainEvent, sampleID, newText) => {
    DB.getInstance()
      .prepare("UPDATE sample SET text=@newText WHERE ID=@sampleID")
      .run({
        newText,
        sampleID,
      });
  }
);

ipcMain.handle(
  FETCH_DATASET_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, datasetID: number) => {
    const speakers = getSpeakersWithSamples(datasetID);
    const ds = DB.getInstance()
      .prepare("SELECT ID, name FROM dataset WHERE ID=@datasetID")
      .get({ datasetID });
    ds.speakers = speakers;
    ds.referencedBy = getReferencedBy(datasetID);
    return ds;
  }
);
