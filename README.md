# VoiceSmith [WIP]

VoiceSmith makes it possible to train and infer on both single and multispeaker models without any coding experience. It fine-tunes a pretty solid text to speech pipeline based on a modified version of [DelightfulTTS](https://arxiv.org/abs/2110.12612) and [UnivNet](https://arxiv.org/abs/2106.07889) on your dataset. Both models were pretrained on a proprietary 5000 speaker dataset. It also provides some tools for dataset preprocessing like automatic text normalization.

If you want to play around with a model trained on a highly emotional emotional 60 speaker dataset using an earlier version of this software [click here](https://colab.research.google.com/drive/1zh6w_TpEAyr_UIojiLmt4ZdYLWeap9mn#scrollTo=vQCA50dao0Mt).

<img src="/.media/hero.png">

| Language | Training + Inference |  Sample Splitting | Text Normalization | # Files Pretrained | # Speakers Pretrained |
| --- | --- | --- | --- | --- | --- |
| English | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | 978283 | 5415 |
| German  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | 380175 | 1034 |

## Requirements

#### Hardware
* OS: Windows (only CPU supported currently) or any Linux based operating system. If you want to run this on macOS you have to follow the steps in build from source in order to create the installer. This is untested since I don't currently own a Mac.
* Graphics: NVIDIA GPU with [CUDA support](https://developer.nvidia.com/cuda-gpus) is highly recommended, you can train on CPU otherwise but it will take days if not weeks.
* RAM: 8GB of RAM, you can try with less but it may not work.

#### Software
* Docker, you can [download it here](https://docs.docker.com/get-docker/). If you are on Linux, it is advised to install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) instead of Docker Desktop, as Docker Desktop likes to make the world complicated.

## How to install

1. Download the latest installer from the [releases page](https://github.com/dunky11/voicesmith/releases).
2. Double click to run the installer.

## How to develop

1. Make sure you have the latest version of [Node.js](https://nodejs.org/) installed
2. Clone the repository

   ```
   git clone https://github.com/dunky11/voicesmith
   ```
3. Install dependencies, this can take a minute

   ```
   cd voicesmith
   npm install
   ```
4. [Click here](https://drive.google.com/drive/folders/15VQgRxGO_Z_RUNMyuJreg9O5Ckcit2vh?usp=sharing), select the folder with the latest version, download all files and place them inside the repositories assets folder.

5. Start the project

   ```
   npm start
   ```
  
## Build from source

1. Follow steps 1 - 4 from above.
2. Run make, this will create a folder named out/make with an installer inside. The installer will be different based on your operating system.
    
    ```
    npm make
    ```
## Architecture

VoiceSmith currently uses a two-stage modified DelightfulTTS and UnivNet pipeline.

<img src="/.media/architecture.png">

 
## Contribute

Show your support by ⭐ the project. Pull requests are always welcome.

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE.md](https://github.com/dunky11/voicesmith/blob/master/LICENSE) file for details.
