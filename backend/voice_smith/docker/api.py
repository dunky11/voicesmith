import docker
from docker.errors import NotFound as NotFoundError, ImageNotFound as ImageNotFoundError
from pathlib import Path
from typing import Any
import multiprocessing as mp
import os
import sys

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
        f"{Path(db_path).parent}:/home/voice_smith/db",
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


def generate_vocab(container: Any, training_run_name: str, workers: int) -> None:
    print("Generating vocabulary ... ")
    run_command(
        container, f"bash ./generate_vocab.sh {training_run_name} {workers}",
    )
    print("Merging vocabulary with presets ...")
    run_command(
        container,
        f"conda run -n voice_smith python merge_lexika.py {training_run_name}",
    )


def align(container: Any, training_run_name: str, workers: int):
    print("Generating alignments ...")
    run_command(
        container, f"bash ./align.sh {training_run_name} {workers}",
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


def container_exec(
    container,
    cmd,
    stdout=True,
    stderr=True,
    stdin=False,
    tty=False,
    privileged=False,
    user="",
    detach=False,
    stream=False,
    socket=False,
    environment=None,
    workdir=None,
):
    """ https://github.com/docker/docker-py/issues/1989
    An enhanced version of #docker.Container.exec_run() which returns an object
    that can be properly inspected for the status of the executed commands.
    """

    exec_id = container.client.api.exec_create(
        container.id,
        cmd,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        tty=tty,
        privileged=privileged,
        user=user,
        environment=environment,
        workdir=workdir,
    )["Id"]

    output = container.client.api.exec_start(
        exec_id, detach=detach, tty=tty, stream=stream, socket=socket
    )

    return ContainerExec(container.client, exec_id, output)


class ContainerExec(object):
    def __init__(self, client, id, output):
        self.client = client
        self.id = id
        self.output = output

    def inspect(self):
        return self.client.api.exec_inspect(self.id)

    def poll(self):
        return self.inspect()["ExitCode"]

    def communicate(self, line_prefix=b""):
        for data in self.output:
            if not data:
                continue
            offset = 0
            print(data)
            while offset < len(data):
                sys.stdout.buffer.write(line_prefix)
                nl = data.find(b"\n", offset)
                if nl >= 0:
                    slice = data[offset : nl + 1]
                    offset = nl + 1
                else:
                    slice = data[offset:]
                    offset += len(slice)
                sys.stdout.buffer.write(slice)
            sys.stdout.flush()
        while self.poll() is None:
            raise RuntimeError("Hm could that really happen?")
        return self.poll()


if __name__ == "__main__":
    client = docker.from_env()
    container = client.containers.get(_CONTAINER_NAME)
    container = client.containers.run(
        _CONTAINER_NAME, tty=True, detach=True, name="TEST"
    )
    ret = container_exec(container, "echo 'hello world!';")
    response_code = ret.communicate()
    print(response_code)
