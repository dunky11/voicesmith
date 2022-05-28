import subprocess
import os


def run_conda_in_shell(cmd: str, environment_name: str) -> bool:
    with subprocess.Popen(
        [f"conda run -n {environment_name} --no-capture-output python {cmd}"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env={**os.environ, "PYTHONNOUSERSITE": "True"},
    ) as process:
        while True:
            output = process.stdout.readline()
            return_code = process.poll()
            if return_code is not None:
                # Process has finished, read rest of the output
                for output in process.stdout.readlines():
                    print(output.strip(), flush=True)
                return return_code == 0
