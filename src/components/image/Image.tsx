import React, { useEffect, useState, useRef } from "react";
const { ipcRenderer } = window.require("electron");
export default function Image({ path, style }: { path: string; style: {} }) {
  const isMounted = useRef(false);
  const [dataUrl, setDataUrl] = useState<string>("");

  const getDataUrl = () => {
    if (path === "") {
      return;
    }
    ipcRenderer.invoke("get-image-data-url", path).then((dataUrl: string) => {
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
