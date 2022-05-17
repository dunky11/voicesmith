import { ipcMain, IpcMainInvokeEvent } from "electron";
import { safeUnlink } from "../utils/files";
import { DB } from "../utils/db";

ipcMain.handle(
  "remove-audios-synth",
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
