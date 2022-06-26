import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { NavigationSettingsInterface } from "../interfaces";

const initialState: NavigationSettingsInterface = {
  isDisabled: false,
};

export const navigationSettingsSlice = createSlice({
  name: "navigationSettings",
  initialState,
  reducers: {
    setNavIsDisabled: (state, action: PayloadAction<boolean>) => {
      state.isDisabled = action.payload;
    },
  },
});

export const { setNavIsDisabled } = navigationSettingsSlice.actions;
export default navigationSettingsSlice.reducer;
