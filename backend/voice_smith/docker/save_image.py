from pathlib import Path
from voice_smith.docker.api import save_image
from voice_smith import ASSETS_PATH

if __name__ == "__main__":
    save_image(str(ASSETS_PATH)) 
