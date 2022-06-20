import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { RunManagerInterface, RunInterface } from "../interfaces";

const initialState: RunManagerInterface = {
  isRunning: false,
  queue: [],
};

export const runManagerSlice = createSlice({
  name: "appInfo",
  initialState,
  reducers: {
    setIsRunning: (state, action: PayloadAction<boolean>) => {
      state.isRunning = action.payload;
    },
    addToQueue: (state, action: PayloadAction<RunInterface>) => {
      state.queue = [action.payload, ...state.queue];
    },
    popFromQueue: (state) => {
      state.queue.shift();
    },
  },
});

export const { setIsRunning, addToQueue, popFromQueue } =
  runManagerSlice.actions;
export default runManagerSlice.reducer;
