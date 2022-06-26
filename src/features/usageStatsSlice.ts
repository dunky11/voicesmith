import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { UsageStatsInterface } from "../interfaces";

const initialState: UsageStatsInterface[] = [];
const USAGE_STATS_MAX_LENGTH = 100;

export const usageStatsSlice = createSlice({
  name: "appInfo",
  initialState,
  reducers: {
    addStats: (state, action: PayloadAction<UsageStatsInterface>) => {
      if (state.length >= USAGE_STATS_MAX_LENGTH) {
        state.shift();
      }
      state.push(action.payload);
    },
  },
});

export const { addStats } = usageStatsSlice.actions;
export default usageStatsSlice.reducer;
