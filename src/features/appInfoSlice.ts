import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { stat } from "original-fs";
import { AppInfoInterface } from "../interfaces";

const initialState: AppInfoInterface = {
  version: null,
  platform: null,
};

export const appInfoSlice = createSlice({
  name: "appInfo",
  initialState,
  reducers: {
    editAppInfo: (state, action: PayloadAction<AppInfoInterface>) => {
      state.platform = action.payload.platform;
      state.version = action.payload.version;
    },
  },
});

export const { editAppInfo } = appInfoSlice.actions;
export default appInfoSlice.reducer;
