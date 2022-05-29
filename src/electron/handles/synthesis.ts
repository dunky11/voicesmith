import { ipcMain, IpcMainInvokeEvent } from "electron";
import path from "path";
import { getAudioSynthDir } from "../utils/globals";
import { safeUnlink } from "../utils/files";
import { DB } from "../utils/db";
import {
  REMOVE_AUDIOS_SYNTH_CHANNEL,
  FETCH_AUDIOS_SYNTH_CHANNEL,
} from "../../channels";

ipcMain.handle(
  FETCH_AUDIOS_SYNTH_CHANNEL.IN,
  async (event: IpcMainInvokeEvent) => {
    const audios = DB.getInstance()
      .prepare(
        `
        SELECT 
        ID, 
        file_name AS fileName, 
        text, 
        speaker_name AS speakerName, 
        model_name AS modelName,
        created_at AS createdAt,
        sampling_rate as samplingRate,
        dur_secs AS durSecs
        FROM audio_synth
        ORDER BY created_at DESC
      `
      )
      .all()
      .map((audio: any) => {
        audio.filePath = path.join(getAudioSynthDir(), audio.fileName);
        delete audio.fileName;
        return audio;
      });
    return audios;
  }
);

ipcMain.handle(
  REMOVE_AUDIOS_SYNTH_CHANNEL.IN,
  async (event: IpcMainInvokeEvent, audios: any[]) => {
    for (const audio of audios) {
      await safeUnlink(audio.filePath);
    }
    const stmt = DB.getInstance().prepare(
      "DELETE FROM audio_synth WHERE ID=@ID"
    );
    const deleteMany = DB.getInstance().transaction((els: any) => {
      for (const el of els) stmt.run(el);
    });
    deleteMany(
      audios.map((audio) => ({
        ID: audio.ID,
      }))
    );
  }
);
