import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RunManagerInterface } from "../interfaces";

const initialState: RunManagerInterface = {
  isRunning: false,
  queue: [],
};

export const runManagerSlice = createSlice({
  name: "appInfo",
  initialState,
  reducers: {
    editRunManager: (state, action: PayloadAction<RunManagerInterface>) => {
      state = action.payload;
    },
  },
});

export const { editRunManager } = runManagerSlice.actions;
export default runManagerSlice.reducer;
