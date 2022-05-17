import { ipcMain, IpcMainInvokeEvent } from "electron";
import { db } from "../utils/db";

ipcMain.handle("fetch-models", async () => {
  const speakersStmt = db.prepare(
    "SELECT name, speaker_id AS speakerID FROM model_speaker WHERE model_id=@ID"
  );
  const models = db
    .prepare(
      "SELECT ID, name, type, description, created_at AS createdAt FROM model"
    )
    .all();
  for (const model of models) {
    model.speakers = speakersStmt.all({ ID: model.ID });
  }
  return models;
});

ipcMain.handle("remove-model", (event: IpcMainInvokeEvent, modelID: number) => {
  db.prepare("DELETE FROM model WHERE ID=@modelID").run({ modelID });
});
