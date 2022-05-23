# VoiceSmith [WIP]

VoiceSmith makes it possible to train and infer on both single and multispeaker models without any coding experience. It fine-tunes a pretty solid text to speech pipeline based on a modified version of [DelightfulTTS](https://arxiv.org/abs/2110.12612) and [UnivNet](https://arxiv.org/abs/2106.07889) on your dataset. Both models were pretrained on a proprietary 5000 speaker dataset. It also provides some tools for data preprocessing like automatic text normalization.

If you want to play around with a model trained on a highly emotional emotional 60 speaker dataset using an earlier version of this software [click here](https://colab.research.google.com/drive/1zh6w_TpEAyr_UIojiLmt4ZdYLWeap9mn#scrollTo=vQCA50dao0Mt).

<img src="/.media/hero.png">

## Getting Started

### Requirements

#### Hardware:
* OS: Windows or any Linux based operating system.
* Graphics: NVIDIA GPU with [CUDA support](https://developer.nvidia.com/cuda-gpus) is heavily recommended, you can train on CPU otherwise but it will take days if not weeks.
* RAM: 8GB of RAM, you can try with less but it may not work.

#### Software:
* Python >=3.7,<3.11, you can [download it here](https://www.python.org/downloads/).
* Docker, you can [download it here](https://docs.docker.com/get-docker/).

### How to install

1. Download the latest installer from the [releases page](https://github.com/dunky11/voicesmith/releases).
2. Double click to run the installer.
