from voice_smith.g2p.dp.phonemizer import Phonemizer
from voice_smith.utils.model import get_param_num

if __name__ == "__main__":

    checkpoint_path = "checkpoints/best_model_no_optim.pt"
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)
    print(get_param_num(phonemizer.predictor.model))    
    text = "young"

    result = phonemizer.phonemise_list([text], lang="en_us")

    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f"{o} {p}")
        tokens = "".join(tokens)
        print(f"{text} | {tokens} | {pred.confidence}")

