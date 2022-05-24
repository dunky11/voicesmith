import docker
from docker.errors import NotFound as NotFoundError, ImageNotFound as ImageNotFoundError
from pathlib import Path
from typing import Any
import multiprocessing as mp
import os

_CONTAINER_NAME = "voice_smith"


def load_image_if_needed() -> None:
    client = docker.from_env()
    client.images.get(_CONTAINER_NAME)


def rerun_container(user_data_path: str, db_path: str) -> Any:
    client = docker.from_env()
    try:
        container = client.containers.get(_CONTAINER_NAME)
        container.remove(force=True)
    except NotFoundError:
        pass

    out_path = "/home/voice_smith/data"
    volumes = [
        f"{user_data_path}:/home/voice_smith/data",
        f"{Path(db_path).parent}:/home/voice_smith/db"
    ]
    container = client.containers.run(
        _CONTAINER_NAME, tty=True, detach=True, name=_CONTAINER_NAME, volumes=volumes
    )
    return container

def get_container() -> Any:
    client = docker.from_env()
    return client.containers.get(_CONTAINER_NAME)


def run_command(container: Any, cmd: str, user="voice_smith") -> None:
    _, stream = container.exec_run(cmd, stream=True, user=user, tty=True)
    for data in stream:
        print(data.decode(), flush=True)


def generate_vocab(container: Any, training_run_name: str) -> None:
    print("Generating vocabulary ... ")
    n_workers = max(1, mp.cpu_count() - 1)
    run_command(
        container,
        f"bash ./generate_vocab.sh {training_run_name} {n_workers}",
    )
    print("Merging vocabulary with presets ...")
    run_command(
        container,
        f"conda run -n voice_smith python merge_lexika.py {training_run_name}",
    )


def align(container: Any, training_run_name: str):
    print("Generating alignments ...")
    n_workers = max(1, mp.cpu_count() - 1)
    run_command(
        container,
        f"bash ./align.sh {training_run_name} {n_workers}",
    )


def text_normalize(container: Any, training_run_name: str, lang: str):
    run_command(
        container,
        f"conda run python text_normalization.py --training_run_id {training_run_name} --run_type textNormalizationRun --lang {lang}",
        user="root",
    )


def save_image(path: str) -> None:
    client = docker.from_env()
    image = client.images.get(_CONTAINER_NAME)
    with open(path, "wb") as f:
        for chunk in image.save():
            f.write(chunk)


def reload_docker(user_data_path: str, db_path: str) -> Any:
    load_image_if_needed()
    container = rerun_container(user_data_path=user_data_path, db_path=db_path)
    return container
