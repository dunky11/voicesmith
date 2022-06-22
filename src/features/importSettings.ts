import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { ImportSettingsInterface } from "../interfaces";

const initialState: ImportSettingsInterface = {
  language: "en",
};

export const importSettingsSlice = createSlice({
  name: "importSettings",
  initialState,
  reducers: {
    editImportSettings: (
      state,
      action: PayloadAction<ImportSettingsInterface>
    ) => {
      state.language = action.payload.language;
    },
  },
});

export const { editImportSettings } = importSettingsSlice.actions;
export default importSettingsSlice.reducer;
