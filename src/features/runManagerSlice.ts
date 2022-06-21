import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { notification } from "antd";
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
    editQueue: (state, action: PayloadAction<RunInterface[]>) => {
      console.log("EDIT QUEUE CALLED");
      state.queue = action.payload;
    },
    addToQueue: (state, action: PayloadAction<RunInterface>) => {
      state.queue = [...state.queue, action.payload];
      if (state.queue.length === 1) {
        state.isRunning = true;
      } else {
        notification["success"]({
          message: "Your run has been added to the queue",
          placement: "top",
        });
      }
    },
    popFromQueue: (state) => {
      const newQueue = [...state.queue];
      newQueue.shift();
      state.queue = newQueue;
    },
  },
});

export const { setIsRunning, addToQueue, popFromQueue, editQueue } =
  runManagerSlice.actions;
export default runManagerSlice.reducer;
