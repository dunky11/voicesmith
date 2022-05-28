import subprocess
import os
import time
import signal
import atexit
import sys


def run_conda_in_shell(
    cmd: str, environment_name: str, stderr_to_stdout: bool, sleep_time: float = 0.25
) -> bool:
    with subprocess.Popen(
        f"conda run -n {environment_name} --no-capture-output {cmd}",
        universal_newlines=True,
        env={**os.environ, "PYTHONNOUSERSITE": "True"},
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if stderr_to_stdout else None,
        preexec_fn=os.setsid,
    ) as process:
        handler_orig = signal.getsignal(signal.SIGTERM)

        def sigterm_handler(sig, frame):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            sys.exit()

        signal.signal(signal.SIGTERM, sigterm_handler)

        while True:
            output = process.stdout.readline()
            if output != "":
                print(output, flush=True)
            return_code = process.poll()
            if return_code is not None:
                signal.signal(signal.SIGTERM, handler_orig)
                return return_code == 0
            time.sleep(sleep_time)
