import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { SpeakerInterface } from "../interfaces";

const initialState: { language: SpeakerInterface["language"] } = {
  language: "en",
};

export const datasetSlice = createSlice({
  name: "dataset",
  initialState,
  reducers: {
    edit: (state, action: PayloadAction<typeof initialState>) => {
      state = action.payload;
    },
  },
});

export const { edit } = datasetSlice.actions;
export default datasetSlice.reducer;
