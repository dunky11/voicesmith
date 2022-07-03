import React, { ReactElement } from "react";
import { TextNormalizationRunInterface } from "../../../interfaces";
import PreprocessingSteps from "../../../components/runs/ProcessingSteps";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: TextNormalizationRunInterface;
}): ReactElement {
  const onBack = () => {
    onStepChange(0);
  };

  const onNext = () => {
    onStepChange(2);
  };

  return (
    <PreprocessingSteps
      onBack={onBack}
      onNext={onNext}
      stage={run.stage}
      run={run}
      logFileName="preprocessing.txt"
      previousStages={["not_started"]}
      currentStages={[
        {
          title: "Text Normalization",
          stageName: "text_normalization",
          progress: run.textNormalizationProgress,
          description: "Normalizing text of each file.",
        },
      ]}
      nextStages={["choose_samples", "finished"]}
    />
  );
}
