import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from typing import Dict, List, Any
from pathlib import Path
import uuid
from torch.jit._serialization import load
from g2p_en import G2p
from voice_smith.utils.tokenization import BertTokenizer, BasicTokenizer
from voice_smith.utils.text_normalization import EnglishTextNormalizer
from voice_smith.sql import get_con
from voice_smith.utils.tools import get_cpu_usage, get_ram_usage, get_disk_usage
from voice_smith.inference import synthesize as synthesize_infer
from voice_smith.utils.loggers import set_stream_location
from voice_smith.utils.audio import save_audio
import sys
from waitress import serve

def get_lexicon(cur: sqlite3.Cursor, model_id: int) -> Dict[str, List[str]]:
    symbols = cur.execute(
        "SELECT word, phonemes FROM lexicon_word WHERE model_id=?", [model_id]
    ).fetchall()
    lexicon: Dict[str, List[str]] = {}
    for word, phonemes in symbols:
        lexicon[word] = phonemes.split(" ")
    return lexicon


def get_symbol2id(cur: sqlite3.Cursor, model_id: int) -> Dict[str, int]:
    symbols = cur.execute(
        "SELECT symbol, symbol_id FROM symbol WHERE model_id=?", [model_id]
    ).fetchall()
    symbol2id: Dict[str, int] = {}
    for symbol, symbol_id in symbols:
        symbol2id[symbol] = symbol_id
    return symbol2id


__model__ = None


def get_model(cur: sqlite3.Cursor, assets_path: str, models_path: str, model_id: int, model_name: str) -> Dict[str, Any]:
    torchscript_dir = Path(models_path) / model_name / "torchscript"
    acoustic_model = load(torchscript_dir / "acoustic_model.pt")
    style_predictor = load(torchscript_dir / "style_predictor.pt")
    vocoder = load(torchscript_dir / "vocoder.pt")
    lexicon = get_lexicon(cur=cur, model_id=model_id)
    symbol2id = get_symbol2id(cur=cur, model_id=model_id)
    text_normalizer = EnglishTextNormalizer()
    bert_tokenizer = BertTokenizer(assets_path)
    tokenizer = BasicTokenizer()

    return {
        "g2p": G2p(),
        "acoustic_model": acoustic_model,
        "style_predictor": style_predictor,
        "vocoder": vocoder,
        "lexicon": lexicon,
        "symbol2id": symbol2id,
        "tokenizer": tokenizer,
        "bert_tokenizer": bert_tokenizer,
        "text_normalizer": text_normalizer,
    }



def run_server(port: int, db_path: str, audio_synth_path: str, models_path: str, assets_path: str):
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return "Backend is running ... "

    @app.route("/is-cuda-available")
    def is_cuda_avaiable():
        return jsonify(available=torch.cuda.is_available())

    @app.route("/get-system-info")
    def get_system_info():
        cpu_usage = get_cpu_usage()
        total_ram, ram_used = get_ram_usage()
        total_disk, disk_used = get_disk_usage()
        return jsonify(
            cpuUsage=cpu_usage,
            totalRam=total_ram,
            ramUsed=ram_used,
            totalDisk=total_disk,
            diskUsed=disk_used,
        )

    @app.route("/open-model", methods=["POST"])
    def open_model():
        """TODO currently doesnt work"""
        return jsonify(success=True)

    @app.route("/close-model", methods=["GET"])
    def close_model():
        __model__ = None
        return jsonify(success=True)

    @app.route("/synthesize", methods=["POST"])
    def synthesize():
        con = get_con(user_data_path)
        cur = con.cursor()
        model_id = request.form.get("modelID", type=int)
        speaker_id = request.form.get("speakerID", type=int)
        text = request.form.get("text")
        talking_speed = request.form.get("talkingSpeed", type=float)

        if (
            model_id == None
            or speaker_id == None
            or text == None
            or talking_speed == None
        ):
            return "Invalid Request.", 400

        row = cur.execute(
            "SELECT name, type FROM model WHERE ID=?",
            (model_id,),
        ).fetchone()
        model_name, type = row

        row = cur.execute(
            "SELECT name FROM model_speaker WHERE speaker_id=? AND model_id=?",
            (speaker_id, model_id),
        ).fetchone()
        speaker_name = row[0]

        __model__ = get_model(cur=cur, assets_path=assets_path, models_path=models_path, model_id=model_id, model_name=model_name)

        logs_dir = Path(models_path) / model_name / "logs"
        audio_dir = Path(audio_synth_path)
        audio_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        set_stream_location(str(logs_dir / "synthesize.txt"))

        audio, sr = synthesize_infer(
            text=text,
            talking_speed=talking_speed,
            speaker_id=speaker_id,
            type=type,
            symbol2id=__model__["symbol2id"],
            lexicon=__model__["lexicon"],
            g2p=__model__["g2p"],
            tokenizer=__model__["tokenizer"],
            acoustic_model=__model__["acoustic_model"],
            text_normalizer=__model__["text_normalizer"],
            vocoder=__model__["vocoder"],
            bert_tokenizer=__model__["bert_tokenizer"],
            style_predictor=__model__["style_predictor"],
        )

        audio_name = f"{uuid.uuid1().hex[:12]}.flac"
        save_audio(str(audio_dir / audio_name), torch.FloatTensor(audio), sr)

        cur.execute(
            """
            INSERT INTO audio_synth (
                file_name,
                text,
                speaker_name,
                model_name,            
                sampling_rate,
                dur_secs
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            [audio_name, text, speaker_name, model_name, sr, audio.shape[0] / sr],
        )
        con.commit()

        return jsonify(success=True)

    serve(app, host="localhost", port=port)

if __name__ == "__main__":
    run_server(port=int(sys.argv[1]), db_path=sys.argv[2], audio_synth_path=sys.argv[3], models_path=sys.argv[4], assets_path=sys.argv[5])
