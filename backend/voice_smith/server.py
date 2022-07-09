import torch
import sqlite3
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Any
from pathlib import Path
import uuid
from torch.jit._serialization import load
from g2p_en import G2p
from waitress import serve
from voice_smith.sql import get_con
from voice_smith.utils.tools import get_cpu_usage, get_ram_usage, get_disk_usage
from voice_smith.inference import synthesize as synthesize_infer
from voice_smith.utils.loggers import set_stream_location
from voice_smith.utils.audio import save_audio
from voice_smith.config.symbols import symbol2id
from voice_smith.config.langs import lang2id
from voice_smith.config.globals import (
    DB_PATH,
    AUDIO_SYNTH_PATH,
    ASSETS_PATH,
    MODELS_PATH,
)
from voice_smith.g2p.dp.utils.model import get_g2p


__model__ = None


def get_model(
    cur: sqlite3.Cursor,
    assets_path: str,
    models_path: str,
    model_id: int,
    model_name: str,
) -> Dict[str, Any]:
    torchscript_dir = Path(models_path) / model_name / "torchscript"
    acoustic_model = load(torchscript_dir / "acoustic_model.pt")
    vocoder = load(torchscript_dir / "vocoder.pt")

    return {
        "g2p": G2p(),
        "acoustic_model": acoustic_model,
        "vocoder": vocoder,
    }


def run_server(port: int):
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
        con = get_con(DB_PATH)
        cur = con.cursor()
        model_id = request.form.get("modelID", type=int)
        speaker_id = request.form.get("speakerID", type=int)
        text = request.form.get("text")
        talking_speed = request.form.get("talkingSpeed", type=float)
        language = request.form.get("language", type=str)

        if (
            model_id == None
            or speaker_id == None
            or text == None
            or talking_speed == None
        ):
            return "Invalid Request.", 400

        row = cur.execute(
            "SELECT name, type FROM model WHERE ID=?", (model_id,),
        ).fetchone()
        model_name, model_type = row

        logs_dir = Path(MODELS_PATH) / model_name / "logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        set_stream_location(str(logs_dir / "synthesize.txt"), log_console=False)

        row = cur.execute(
            "SELECT name FROM model_speaker WHERE speaker_id=? AND model_id=?",
            (speaker_id, model_id),
        ).fetchone()
        speaker_name = row[0]

        __model__ = get_model(
            cur=cur,
            assets_path=ASSETS_PATH,
            models_path=MODELS_PATH,
            model_id=model_id,
            model_name=model_name,
        )

        audio_dir = Path(AUDIO_SYNTH_PATH)
        audio_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio, sr = synthesize_infer(
            text=text,
            lang=language,
            talking_speed=talking_speed,
            speaker_id=speaker_id,
            model_type=model_type,
            symbol2id=symbol2id,
            lang2id=lang2id,
            g2p=get_g2p(assets_path=ASSETS_PATH, device=device),
            acoustic_model=__model__["acoustic_model"],
            vocoder=__model__["vocoder"],
            device=device,
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

    serve(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    run_server(port=int(args.port))
