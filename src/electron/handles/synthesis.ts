import { ipcMain, IpcMainInvokeEvent } from "electron";
import { safeUnlink } from "../utils/files";
import { db } from "../utils/db";

ipcMain.handle(
  "remove-audios-synth",
  async (event: IpcMainInvokeEvent, audios: any[]) => {
    for (const audio of audios) {
      await safeUnlink(audio.filePath);
    }
    const stmt = db.prepare("DELETE FROM audio_synth WHERE ID=@ID");
    const deleteMany = db.transaction((els) => {
      for (const el of els) stmt.run(el);
    });
    deleteMany(
      audios.map((audio) => ({
        ID: audio.ID,
      }))
    );
  }
);
