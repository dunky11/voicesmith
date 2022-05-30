from pathlib import Path

from voice_smith.g2p.dp.model.model import load_checkpoint, ModelType, create_model
from voice_smith.g2p.dp.preprocessing.text import Preprocessor
from voice_smith.g2p.dp.training.trainer import Trainer
from voice_smith.g2p.dp.utils.logging import get_logger

logger = get_logger(__name__)


def train(
    config: str,
    name: str,
    checkpoint_file: str = None,
) -> None:
    """
    Runs training of a transformer model.

    Args:
      config_file (str): Path to the config.yaml that stores all necessary parameters.
      checkpoint_file (str, optional): Path to a model checkpoint to resume training for (e.g. latest_model.pt)

    Returns:
        None: The model checkpoints are stored in a folder provided by the config.

    """

    if checkpoint_file is not None:
        logger.info(f'Restoring model from checkpoint: {checkpoint_file}')
        model, checkpoint = load_checkpoint(checkpoint_file)
        model.train()
        step = checkpoint['step']
        logger.info(f'Loaded model with step: {step}')
        for key, val in config['training'].items():
            val_orig = checkpoint['config']['training'][key]
            if val_orig != val:
                logger.info(f'Overwriting training param: {key} {val_orig} --> {val}')
                checkpoint['config']['training'][key] = val
        config = checkpoint['config']
        model_type = config['model']['type']
        model_type = ModelType(model_type)
    else:
        logger.info('Initializing new model from config...')
        preprocessor = Preprocessor.from_config(config)
        model_type = config['model']['type']
        model_type = ModelType(model_type)
        model = create_model(model_type, config=config)
        checkpoint = {
            'config': config,
        }

    if "preprocessor" in checkpoint.keys():
        del checkpoint["preprocessor"]

    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    logger.info(f'Checkpoints will be stored at {checkpoint_dir.absolute()}')
    loss_type = 'cross_entropy' if model_type.is_autoregressive() else 'ctc'
    trainer = Trainer(checkpoint_dir=checkpoint_dir, loss_type=loss_type, name=name, config=config)
    trainer.train(
        model=model,
        checkpoint=checkpoint,
        store_phoneme_dict_in_model=config['training']['store_phoneme_dict_in_model']
    )