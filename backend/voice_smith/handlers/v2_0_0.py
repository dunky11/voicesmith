import torch
import os
import json
from typing import List


def strip_invalid_symbols(text: str, pad_symbol: str, valid_symbols: List[str]) -> str:
    new_token = ""
    for char in text.lower():
        if char in valid_symbols and char != pad_symbol:
            new_token += char
    return new_token


class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.manifest = None
        self.initialized = False
        self.g2p: torch.jit.ScriptModule = None
        self.acoustic: torch.jit.ScriptModule = None
        self.vocoder: torch.jit.ScriptModule = None
        self.device: torch.device = None
        self.phone2id: Dict[str, int] = None
        self.speaker2id: Dict[str, int] = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # Read model serialize/pt file
        g2p_file = self.manifest["model"]["g2pFile"]
        acoustic_file = self.manifest["model"]["acousticFile"]
        vocoder_file = self.manifest["model"]["vocoderFile"]
        phone2id_file = self.manifest["model"]["phone2idFile"]
        speaker2id_file = self.manifest["model"]["speaker2idFile"]

        self.g2p = torch.jit.load(os.path.join(model_dir, g2p_file))
        self.acoustic = torch.jit.load(os.path.join(model_dir, acoustic_file))
        self.vocoder = torch.jit.load(os.path.join(model_dir, vocoder_file))

        with open(os.path.join(model_dir, phone2id_file), "r", encoding="utf-8") as f:
            self.phone2id = json.load(f)

        with open(os.path.join(model_dir, speaker2id_file), "r", encoding="utf-8") as f:
            self.speaker2id = json.load(f)

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        if data.get("useSSML"):
            raise Exception("SSML is not implemented yet", 400)
        else:
            text = data.get("text")
            lang = data.get("lang")
            speed = data.get("speed")
            pitch = data.get("pitch")

            text = text.strip()
            text = remove_cont_whitespaces(text)
            word_tokenizer = WordTokenizer(lang=lang, remove_punct=False)
            sentence_tokenizer = SentenceTokenizer(lang=lang)

            waves = []
            for sentence in sentence_tokenizer.tokenize(text):
                symbol_ids = []
                sentence = text_normalizer(sentence)
                for word in word_tokenizer.tokenize(sentence):
                    word = word.lower()
                    if word.strip() == "":
                        continue
                    elif word in [".", "?", "!"]:
                        symbol_ids.append(self.symbol2id[word])
                    elif word in [",", ";"]:
                        symbol_ids.append(self.symbol2id["SILENCE"])
                        """
                        # This can be done inside torchscript?
                        elif word in lexicon:
                            for phone in lexicon[word]:
                                symbol_ids.append(symbol2id[phone])
                            symbol_ids.append(symbol2id["BLANK"])
                        """
                    else:
                        # TODO batch this
                        for phone in self.g2p([word], [lang]):
                            symbol_ids.append(symbol2id[phone])
                        symbol_ids.append(symbol2id["BLANK"])

                sentence_style = " ".join(style_sentences)

                encoding = bert_tokenizer([sentence_style])

                symbol_ids = torch.LongTensor([symbol_ids])
                speaker_ids = torch.LongTensor([speaker_id])

                with torch.no_grad():
                    mel = acoustic_model(symbol_ids, speaker_ids, 1.0, talking_speed,)
                    wave = vocoder(mel)
                    waves.append(wave.view(-1))

            wave_cat = torch.cat(waves).numpy()

        return preprocessed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
