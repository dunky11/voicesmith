import React, { ReactElement } from "react";
import PreprocessingSteps from "../../../components/runs/ProcessingSteps";
import { CleaningRunInterface } from "../../../interfaces";

export default function Configuration({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: CleaningRunInterface;
}): ReactElement {
  const onBack = () => {
    onStepChange(0);
  };

  const onNext = () => {
    onStepChange(2);
  };

  return (
    <PreprocessingSteps
      title="Configuration"
      onBack={onBack}
      onNext={onNext}
      stage={run.stage}
      run={run}
      logFileName="preprocessing.txt"
      previousStages={["not_started"]}
      currentStages={[
        {
          title: "Copy Files",
          stageName: "copying_files",
          progress: run.copyingFilesProgress,
          description: "Copy audio files into the correct folder.",
        },
        {
          title: "Transcribe",
          stageName: "transcribe",
          progress: run.transcriptionProgress,
          description: "Transcribe audio to calculate sample quality score.",
        },
      ]}
      nextStages={["choose_samples", "apply_changes", "finished"]}
    />
  );
}
