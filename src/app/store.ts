import { configureStore } from "@reduxjs/toolkit";
import appInfo from "../features/appInfoSlice";
import runManager from "../features/runManagerSlice";
import useStats from "../features/usageStatsSlice";
import importSettings from "../features/importSettings";
import navigationSettings from "../features/navigationSettingsSlice";
import documentationManager from "../features/documentationManagerSlice";

export const store = configureStore({
  reducer: {
    appInfo,
    runManager,
    useStats,
    importSettings,
    navigationSettings,
    documentationManager,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
