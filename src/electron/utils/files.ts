import fs from "fs-extra";
import fsNative from "fs";
const fsPromises = fsNative.promises;

export const exists = (file: string) => {
  return fsPromises
    .access(file, fs.constants.F_OK)
    .then(() => true)
    .catch(() => false);
};

export const copyDir = (src: string, dest: string) => {
  return new Promise((resolve, reject) => {
    fs.copy(src, dest)
      .then(() => {
        resolve(null);
      })
      .catch((err) => {
        reject(err);
      });
  });
};

export const safeUnlink = async (path: string) => {
  try {
    await fsPromises.unlink(path);
  } catch (err) {
    if (err.code === "ENOENT") {
      return;
    }
    throw err;
  }
};

export const safeMkdir = async (path: string) => {
  await fsPromises.mkdir(path, { recursive: true });
};

export const safeRmDir = async (path: string) => {
  try {
    await fsPromises.rm(path, { recursive: true, force: true });
  } catch (err) {
    if (err.code === "ENOENT") {
      return;
    }
    throw err;
  }
};
