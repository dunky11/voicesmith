import { ipcMain, IpcMainInvokeEvent } from "electron";
import { DB } from "../utils/db";

ipcMain.handle("fetch-models", async () => {
  const speakersStmt = DB.getInstance().prepare(
    "SELECT name, speaker_id AS speakerID FROM model_speaker WHERE model_id=@ID"
  );
  const models = DB.getInstance()
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
  DB.getInstance()
    .prepare("DELETE FROM model WHERE ID=@modelID")
    .run({ modelID });
});
