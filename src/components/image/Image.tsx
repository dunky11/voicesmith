import React, { useEffect, useState, useRef, ReactElement } from "react";
import { GET_IMAGE_DATA_URL_CHANNEL } from "../../channels";
const { ipcRenderer } = window.require("electron");

export default function Image({
  path,
  style,
}: {
  path: string;
  style: {};
}): ReactElement {
  const isMounted = useRef(false);
  const [dataUrl, setDataUrl] = useState<string>("");

  const getDataUrl = () => {
    if (path === "") {
      return;
    }
    ipcRenderer
      .invoke(GET_IMAGE_DATA_URL_CHANNEL.IN, path)
      .then((dataUrl: string) => {
        if (!isMounted.current) {
          return;
        }
        setDataUrl(dataUrl);
      });
  };

  useEffect(() => {
    getDataUrl();
  }, [path]);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  return dataUrl ? <img style={style} src={dataUrl} /> : <></>;
}

Image.defaultProps = {
  style: {},
};
