from voice_smith.utils.shell import run_conda_in_shell


def generate_vocab(environment_name: str, in_path: str, out_path: str, n_workers: int):
    cmd = f"mfa g2p --clean -j {n_workers} english_us_arpa {in_path} {out_path}"
    success = run_conda_in_shell(cmd, environment_name)
    if not success:
        raise Exception("An error occured in generate_vocab() ...")
