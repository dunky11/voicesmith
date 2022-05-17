import sys
from voice_smith.server import run_server
from voice_smith.training_run import continue_training_run
from voice_smith.cleaning_run import continue_cleaning_run
from voice_smith.text_normalization_run import continue_text_normalization_run

if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Please provide the type of script to run"
    name = str(sys.argv[1])
    if name == "server":
        assert (
            len(sys.argv) == 3
        ), "If you wan't to run the server please specify a port"
        port = int(sys.argv[2])
        run_server(port=port)
    elif name == "training_run":
        assert (
            len(sys.argv) == 3
        ), "If you wan't to continue the training run, please provide it's ID ..."
        ID = int(sys.argv[2])
        continue_training_run(training_run_id=ID)
    elif name == "cleaning_run":
        assert (
            len(sys.argv) == 3
        ), "If you wan't to continue the cleaning run, please provide it's ID ..."
        ID = int(sys.argv[2])
        continue_cleaning_run(cleaning_run_id=ID)
    elif name == "text_normalization_run":
        assert (
            len(sys.argv) == 3
        ), "If you wan't to continue the text normalization run, please provide it's ID ..."
        ID = int(sys.argv[2])
        continue_text_normalization_run(ID=ID)
    else:
        raise Exception(f"'{name}' is not a valid script name to run ...")
