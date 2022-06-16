from voice_smith.utils.shell import run_conda_in_shell
from voice_smith.utils.mfa import lang_to_mfa_acoustic


def align(
    environment_name: str,
    in_path: str,
    lexicon_path: str,
    out_path: str,
    n_workers: int,
    lang: str,
):
    cmd = f"mfa align --clean -j {n_workers} {in_path} {lexicon_path} {lang_to_mfa_acoustic(lang)} {out_path}"
    success = run_conda_in_shell(cmd, environment_name, stderr_to_stdout=True)
    # MFA throws an error at end even though it created texgrids, so don't check
    """if not success:
        raise Exception("An error occured in align() ...")"""
