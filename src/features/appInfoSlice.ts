import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { AppInfoInterface } from "../interfaces";

const initialState: AppInfoInterface = null;

export const appInfoSlice = createSlice({
  name: "appInfo",
  initialState,
  reducers: {
    editAppInfo: (state, action: PayloadAction<AppInfoInterface>) => {
      state = action.payload;
    },
  },
});

export const { editAppInfo } = appInfoSlice.actions;
export default appInfoSlice.reducer;
