import os
from pathlib import Path
import glob


def should_import_modules():
    in_dir = str(Path(os.path.dirname(__file__)) / "voice_smith")
    for module in glob.iglob(f"{in_dir}/**/*.py"):
        if module == "__init__.py" or module[-3:] != ".py":
            continue
        __import__(module[:-3].replace("/", "."), locals(), globals())


if __name__ == "__main__":
    should_import_modules()
