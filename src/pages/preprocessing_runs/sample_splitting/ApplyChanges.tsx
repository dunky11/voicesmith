import React, { ReactElement } from "react";
import ProcessingSteps from "../../../components/runs/ProcessingSteps";
import { SampleSplittingRunInterface } from "../../../interfaces";

export default function ApplyChanges({
  onStepChange,
  run,
}: {
  onStepChange: (current: number) => void;
  run: SampleSplittingRunInterface | null;
}): ReactElement {
  const onBack = () => {
    onStepChange(2);
  };

  return (
    <ProcessingSteps
      title="Apply Changes"
      docsUrl="/usage/sample-splitting/apply-changes"
      onBack={onBack}
      onNext={null}
      stage={run.stage}
      run={run}
      logFileName="apply_changes.txt"
      previousStages={[
        "not_started",
        "copying_files",
        "gen_vocab",
        "gen_alignments",
        "creating_splits",
        "choose_samples",
      ]}
      currentStages={[
        {
          title: "Apply Changes",
          stageName: "apply_changes",
          progress: run.applyingChangesProgress,
          description: "Splitting dataset by the sentence boundaries.",
        },
      ]}
      nextStages={["finished"]}
    />
  );
}
