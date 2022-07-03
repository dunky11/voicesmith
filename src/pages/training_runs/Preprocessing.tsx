import React, { ReactElement } from "react";
import PreprocessingSteps from "../../components/runs/ProcessingSteps";
import { TrainingRunInterface } from "../../interfaces";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (step: number) => void;
  run: TrainingRunInterface;
}): ReactElement {
  const onBack = () => {
    onStepChange(1);
  };

  const onNext = () => {
    onStepChange(3);
  };

  return (
    <PreprocessingSteps
      onBack={onBack}
      onNext={onNext}
      stage={run.preprocessingStage}
      run={run}
      logFileName="preprocessing.txt"
      previousStages={["not_started"]}
      currentStages={[
        {
          title: "Copy Files",
          stageName: "copying_files",
          progress: run.preprocessingCopyingFilesProgress,
          description: "Copy audio files into the correct folder.",
        },
        {
          title: "Generate Vocabulary",
          stageName: "gen_vocab",
          progress: run.preprocessingGenVocabProgress,
          description: "Generate phonemes for each word.",
        },
        {
          title: "Generate Alignments",
          stageName: "gen_alignments",
          progress: run.preprocessingGenAlignProgress,
          description: "Generate timestamps for each phoneme.",
        },
        {
          title: "Extract Data",
          stageName: "extract_data",
          progress: run.preprocessingExtractDataProgress,
          description:
            "Resample audio and extract pitch information and mel-spectrograms.",
        },
      ]}
      nextStages={["finished"]}
    />
  );
}
