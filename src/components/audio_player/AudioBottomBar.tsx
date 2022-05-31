import React, { useEffect, useState, useRef, ReactElement } from "react";
import { Card, Typography } from "antd";
import { PlayCircleFilled, PauseCircleFilled } from "@ant-design/icons";
import { createUseStyles } from "react-jss";

const useStyles = createUseStyles({
  wrapper: {
    height: 64,
    left: 0,
    position: "fixed",
    bottom: 0,
    width: "100%",
  },
  card: {
    width: "calc(100% - 40px)",
    height: "100%",
    marginLeft: 40,
    borderTopLeftRadius: 32,
    borderBottomLeftRadius: 32,
  },
  mainActionIcon: {
    fontSize: 40,
    transition: "opacity 0.3s",
    cursor: "pointer",
    "&:hover": {
      opacity: 0.9,
    },
  },
  audioBarWrapper: {
    marginLeft: 24,
    marginRight: 0,
    flexGrow: 2,
    height: 24,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    "&:hover $audioBar": {
      opacity: 0.8,
    },
  },
  audioBar: {
    backgroundColor: "#d0d8e6",
    height: 5,
    borderRadius: 16,
    width: "100%",
    position: "relative",
    transition: "opacity 0.3s",
  },
  audioBarInner: {
    position: "absolute",
    height: "100%",
    borderRadius: 16,
    backgroundColor: "rgb(22, 22, 25)",
  },
  audioTime: {
    marginLeft: 24,
    marginBottom: 0,
  },
});

const prettyPrintDuration = (duration: number) => {
  let rounded = duration.toFixed(2);
  if (parseFloat(rounded) < 10.0) {
    rounded = `0${rounded}`;
  }
  return rounded;
};

export default function AudioBottomBar({
  src,
  playFuncRef,
}: {
  src: string | null;
  playFuncRef: React.MutableRefObject<null | (() => void)>;
}): ReactElement {
  const classes = useStyles();
  const isMounted = useRef<boolean>(false);
  const playbackIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  const playAudio = () => {
    if (audioRef.current === null) {
      return;
    }
    audioRef.current.play();
  };

  const pauseAudio = () => {
    if (audioRef.current === null) {
      return;
    }
    audioRef.current.pause();
  };

  const addAudioListeners = () => {
    if (audioRef.current === null) {
      throw new Error("audioRef cannot be null ...");
    }
    audioRef.current.addEventListener("play", () => {
      if (!isMounted.current) {
        return;
      }
      if (playbackIntervalRef.current != null) {
        clearInterval(playbackIntervalRef.current);
      }
      playbackIntervalRef.current = setInterval(() => {
        setProgress((progress) => progress + 0.02);
      }, 20);
      setIsPlaying(true);
    });
    audioRef.current.addEventListener("pause", () => {
      if (!isMounted.current) {
        return;
      }
      setIsPlaying(false);
      if (playbackIntervalRef.current != null) {
        clearInterval(playbackIntervalRef.current);
      }
    });
    audioRef.current.addEventListener("ended", () => {
      if (!isMounted.current) {
        return;
      }
      if (playbackIntervalRef.current != null) {
        clearInterval(playbackIntervalRef.current);
      }
      setProgress(0.0);
    });
    audioRef.current.addEventListener("loadeddata", () => {
      if (audioRef.current === null || !isMounted.current) {
        return;
      }
      setDuration(audioRef.current.duration);
    });
  };

  const onAudioBarClick = (event: React.MouseEvent) => {
    if (!isMounted.current || duration === 0 || audioRef.current === null) {
      return;
    }
    const target = event.target;
    // @ts-ignore
    const rect = target.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const width = rect.width;
    const perc = mouseX / width;
    if (isPlaying) {
      setIsPlaying(false);
    }
    setProgress(duration * perc);
    audioRef.current.currentTime = duration * perc;
  };

  useEffect(() => {
    if (src === null) {
      return;
    }
    setProgress(0);
    if (audioRef.current !== null) {
      audioRef.current.currentTime = 0;
    }
    playAudio();
  }, [src]);

  useEffect(() => {
    isMounted.current = true;
    playFuncRef.current = playAudio;
    addAudioListeners();
    return () => {
      playFuncRef.current = null;
      isMounted.current = false;
    };
  }, []);

  return (
    <div className={classes.wrapper}>
      <audio
        src={src === null ? undefined : src}
        ref={(node) => {
          audioRef.current = node;
        }}
      ></audio>
      <Card
        className={classes.card}
        bodyStyle={{
          margin: 0,
          padding: 0,
          display: "flex",
          alignItems: "center",
          paddingLeft: 32,
          paddingRight: 32,
          height: "100%",
        }}
      >
        <div>
          {isPlaying ? (
            <PauseCircleFilled
              onClick={pauseAudio}
              className={classes.mainActionIcon}
            ></PauseCircleFilled>
          ) : (
            <PlayCircleFilled
              onClick={playAudio}
              className={classes.mainActionIcon}
            />
          )}
        </div>
        <div>
          <Typography.Text strong className={classes.audioTime}>
            {prettyPrintDuration(progress)} / {prettyPrintDuration(duration)}
          </Typography.Text>
        </div>
        <div className={classes.audioBarWrapper} onClick={onAudioBarClick}>
          <div className={classes.audioBar}>
            <div
              className={classes.audioBarInner}
              style={{
                width:
                  duration === 0 ? "0%" : `${(progress / duration) * 100}%`,
              }}
            ></div>
          </div>
        </div>
      </Card>
    </div>
  );
}
