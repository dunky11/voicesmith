from voice_smith.utils.shell import run_conda_in_shell


def align(
    environment_name: str,
    in_path: str,
    lexicon_path: str,
    out_path: str,
    n_workers: int,
):
    cmd = f"mfa align --clean -j {n_workers} {in_path} {lexicon_path} english_us_arpa {out_path}"
    success = run_conda_in_shell(cmd, environment_name, stderr_to_stdout=True)
    if not success:
        raise Exception("An error occured in align() ...")
