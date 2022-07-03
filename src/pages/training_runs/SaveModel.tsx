import React, { ReactElement } from "react";
import PreprocessingSteps from "../../components/runs/ProcessingSteps";
import { TrainingRunInterface } from "../../interfaces";

export default function SaveModel({
  onStepChange,
  run,
}: {
  onStepChange: (step: number) => void;
  run: TrainingRunInterface;
}): ReactElement {
  const onBack = () => {
    onStepChange(5);
  };

  return (
    <PreprocessingSteps
      onBack={onBack}
      onNext={null}
      stage={run.stage}
      run={run}
      logFileName="save_model.txt"
      previousStages={[
        "not_started",
        "preprocessing",
        "acoustic_fine_tuning",
        "ground_truth_alignment",
        "vocoder_fine_tuning",
      ]}
      currentStages={[
        {
          title: "Save model",
          stageName: "save_model",
          progress: null,
          description: "",
        },
      ]}
      nextStages={["finished"]}
    />
  );
}
