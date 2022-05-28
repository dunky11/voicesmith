import torch
from pathlib import Path
import json
from torch.utils.data import DataLoader
from typing import Dict, Literal, Optional, Union
from voice_smith.utils.model import get_acoustic_models
from voice_smith.utils.tools import to_device, iter_logger, get_embeddings
from voice_smith.acoustic_training import get_data_loaders
from voice_smith.config.acoustic_model_config import acoustic_model_config
from voice_smith.config.acoustic_fine_tuning_config import acoustic_fine_tuning_config
from voice_smith.config.preprocess_config import preprocess_config
from voice_smith.utils.loggers import Logger
from voice_smith.model.acoustic_model import AcousticModel


def save_gta(
    db_id: int,
    table_name: str,
    gen: AcousticModel,
    style_predictor: torch.nn.Module,
    loader: DataLoader,
    id2speaker: Dict[int, str],
    device: torch.device,
    data_dir: str,
    step: Union[Literal["train"], Literal["val"]],
    logger: Optional[Logger],
    log_every: int = 50,
):
    gen.eval()
    style_predictor.eval()
    output_mels_gta_dir = Path(data_dir) / "data_gta"

    def callback(index: int):
        if logger is None:
            return
        if index % log_every == 0:
            if step == "train":
                progress = (index / len(loader)) * (4 / 5)
            else:
                progress = (4 / 5) + (index / len(loader)) / 5
            logger.query(
                f"UPDATE {table_name} SET ground_truth_alignment_progress=? WHERE id=?",
                [progress, db_id],
            )

    for batchs in iter_logger(loader, cb=callback, total=len(loader)):
        for batch in batchs:
            batch = to_device(batch, device)
            (
                ids,
                raw_texts,
                speakers,
                speaker_names,
                texts,
                src_lens,
                mels,
                pitches,
                durations,
                mel_lens,
                token_ids,
                attention_masks,
            ) = batch
            with torch.no_grad():
                style_embeds_pred = style_predictor(token_ids, attention_masks)
                outputs = gen.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=src_lens,
                    mels=mels,
                    mel_lens=mel_lens,
                    style_embeds_pred=style_embeds_pred,
                    attention_mask=attention_masks,
                    pitches=pitches,
                    durations=durations,
                    use_ground_truth=False,
                )
                y_pred = outputs["y_pred"]
            for basename, speaker_id, mel_pred, mel_len, mel in zip(
                ids, speakers, y_pred, mel_lens, mels
            ):
                speaker_name = id2speaker[int(speaker_id.item())]
                (output_mels_gta_dir / speaker_name).mkdir(exist_ok=True, parents=True)
                mel_pred = mel_pred[:, :mel_len]
                torch.save(
                    {"mel": mel_pred.cpu()},
                    output_mels_gta_dir / speaker_name / f"{basename}.pt",
                )


def ground_truth_alignment(
    db_id: int,
    table_name: str,
    training_run_name: str,
    batch_size: int,
    group_size: int,
    checkpoint_acoustic: str,
    checkpoint_style: str,
    device: torch.device,
    logger: Optional[Logger],
    assets_path: str,
    training_runs_path: str,
    log_every: int = 200,
):
    print("Generating ground truth aligned data ... \n")
    # TODO change group size automatically
    data_path = Path(training_runs_path) / str(training_run_name) / "data"
    group_size = 5
    train_loader, eval_loader = get_data_loaders(
        batch_size=batch_size,
        group_size=group_size,
        data_path=str(data_path),
        assets_path=assets_path,
    )
    with open(data_path / "speakers.json", "r", encoding="utf-8") as f:
        speakers = json.load(f)

    id2speaker = {speakers[key]: key for key in speakers.keys()}

    gen, style_predictor, _, _ = get_acoustic_models(
        checkpoint_acoustic=checkpoint_acoustic,
        checkpoint_style=checkpoint_style,
        data_path=str(data_path),
        train_config=acoustic_fine_tuning_config,
        preprocess_config=preprocess_config,
        model_config=acoustic_model_config,
        fine_tuning=True,
        device=device,
        reset=False,
        assets_path=assets_path,
    )

    print("Generating GTA for training set ... \n")
    save_gta(
        db_id=db_id,
        table_name=table_name,
        gen=gen,
        style_predictor=style_predictor,
        loader=train_loader,
        id2speaker=id2speaker,
        device=device,
        data_dir=str(data_path),
        logger=logger,
        log_every=log_every,
        step="train",
    )

    print("Generating GTA for validation set ... \n")
    save_gta(
        db_id=db_id,
        table_name=table_name,
        gen=gen,
        style_predictor=style_predictor,
        loader=eval_loader,
        id2speaker=id2speaker,
        device=device,
        data_dir=str(data_path),
        logger=logger,
        log_every=log_every,
        step="val",
    )

    logger.query(
        f"UPDATE {table_name} SET ground_truth_alignment_progress=? WHERE id=?",
        [1.0, db_id],
    )


if __name__ == "__main__":

    class NoLogger:
        def query(self, a, b):
            pass

    ground_truth_alignment(
        db_id=None,
        table_name=None,
        training_run_name="pretraining_two_stage_ac",
        batch_size=16,
        group_size=3,
        checkpoint_acoustic=Path(".")
        / "training_runs"
        / "pretraining_two_stage_ac"
        / "ckpt"
        / "acoustic"
        / "acoustic_500000.pt",
        checkpoint_style=Path(".")
        / "training_runs"
        / "pretraining_two_stage_ac"
        / "ckpt"
        / "acoustic"
        / "style_500000.pt",
        device=torch.device("cuda"),
        logger=NoLogger(),
    )
