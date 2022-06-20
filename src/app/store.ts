import { configureStore } from "@reduxjs/toolkit";
import appInfo from "../features/appInfoSlice";
import runManager from "../features/runManagerSlice";

export const store = configureStore({
  reducer: { appInfo, runManager },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
