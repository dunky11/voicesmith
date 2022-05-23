# VoiceSmith [WIP]

VoiceSmith makes it possible to train and infer on both single and multispeaker models without any coding experience. It fine-tunes a pretty solid text to speech pipeline based on a modified version of [DelightfulTTS](https://arxiv.org/abs/2110.12612) and [UnivNet](https://arxiv.org/abs/2106.07889) on your dataset. Both models were pretrained on a proprietary 5000 speaker dataset. It also provides some tools for data preprocessing like automatic text normalization.

If you want to play around with a model trained on a highly emotional emotional 60 speaker dataset using an earlier version of this software [click here](https://colab.research.google.com/drive/1zh6w_TpEAyr_UIojiLmt4ZdYLWeap9mn#scrollTo=vQCA50dao0Mt).

<img src="/.media/hero.png">

## Requirements

#### Hardware
* OS: Windows or any Linux based operating system.
* Graphics: NVIDIA GPU with [CUDA support](https://developer.nvidia.com/cuda-gpus) is heavily recommended, you can train on CPU otherwise but it will take days if not weeks.
* RAM: 8GB of RAM, you can try with less but it may not work.

#### Software
* Python >=3.7,<3.11, you can [download it here](https://www.python.org/downloads/).
* Docker, you can [download it here](https://docs.docker.com/get-docker/).

## How to install

1. Download the latest installer from the [releases page](https://github.com/dunky11/voicesmith/releases).
2. Double click to run the installer.

## How to develope

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
4. Start the project

   ```
   npm start
   ```
  
## Build from source

1. Follow steps 1 - 3 from above.
2. Run make, his will build an installer dependent on your operating system. The installer will be placed inside the out folder in the projects root.
    
    ```
    npm make
    ```
    
## Contribute

Show your support by ⭐ the project. Pull requests are always welcome.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](https://github.com/dunky11/voicesmith/blob/master/LICENSE) file for details.
