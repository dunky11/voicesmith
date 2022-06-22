import { configureStore } from "@reduxjs/toolkit";
import appInfo from "../features/appInfoSlice";
import runManager from "../features/runManagerSlice";
import useStats from "../features/usageStatsSlice";
import dataset from "../features/datasetSlice";

export const store = configureStore({
  reducer: { appInfo, runManager, useStats, dataset },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
