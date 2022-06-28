import { createSlice, PayloadAction } from "@reduxjs/toolkit";
import { DocumentationManagerInterface } from "../interfaces";

const initialState: DocumentationManagerInterface = {
  isOpen: false,
  route: "",
};

export const documentationManagerSlice = createSlice({
  name: "documentationManager",
  initialState,
  reducers: {
    setIsOpen: (state, action: PayloadAction<boolean>) => {
      state.isOpen = action.payload;
    },
    pushRoute: (state, action: PayloadAction<string>) => {
      state.route = action.payload;
    },
  },
});

export const { setIsOpen, pushRoute } = documentationManagerSlice.actions;
export default documentationManagerSlice.reducer;
