import React, { ReactElement } from "react";
import PreprocessingSteps from "../../../components/runs/ProcessingSteps";
import { SampleSplittingRunInterface } from "../../../interfaces";

export default function Preprocessing({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface;
}): ReactElement {
  const onBack = () => {
    onStepChange(0);
  };

  const onNext = () => {
    onStepChange(2);
  };

  return (
    <PreprocessingSteps
      title="Preprocessing"
      docsUrl="/usage/sample-splitting#preprocessing"
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
          title: "Generate Vocabulary",
          stageName: "gen_vocab",
          progress: run.genVocabProgress,
          description: "Generate phonemes for each word.",
        },
        {
          title: "Generate Alignments",
          stageName: "gen_alignments",
          progress: run.genAlignProgress,
          description: "Generate timestamps for each phoneme.",
        },
        {
          title: "Creating Splits",
          stageName: "creating_splits",
          progress: run.creatingSplitsProgress,
          description: "Splitting audios by sentence boundaries.",
        },
      ]}
      nextStages={["choose_samples", "apply_changes", "finished"]}
    />
  );
}
