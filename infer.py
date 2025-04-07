import sys
import os
import torch
import yaml
import soundfile as sf
import torchaudio
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list

# ill make batch infer later but for now just one file each time

frame_duration = 0.02  # ~20ms per frame

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def infer_audio(
    audio_path: str,
    config_path: str = "config.yaml",
    checkpoint_path: str = "best_model.pt",
    output_lab_path: str = None,
    device: str = "cuda"
):
    """
    audio_path: path to the .wav file to transcribe
    config_path: path to the config.yaml file
    checkpoint_path: path to the trained model checkpoint (.pt file)
    output_lab_path: save predicted segments to this .lab file
    device: "cuda" or "cpu". Make sure CUDA is available if using "cuda"
    return: A list of predicted segments, where each segment is (start_time, end_time, phoneme)
    """
    config = load_config(config_path)

    label_list = load_phoneme_list(config["data"]["phoneme_set"])

    model = BIOPhonemeTagger(config, label_list)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    audio, sr = sf.read(audio_path)

    if sr != config["data"]["sample_rate"]:
        audio = torchaudio.functional.resample(
            torch.tensor(audio), orig_freq=sr, new_freq=config["data"]["sample_rate"]
        ).numpy()
        sr = config["data"]["sample_rate"]

    if len(audio) > 0:
        audio = audio / (max(abs(audio)) + 1e-8)

    input_values = torch.tensor(audio, dtype=torch.float32).to(device)

    with torch.no_grad():
        # the model expects shape [batch_size=1, T] inside forward
        # logits shape: [1, frames, num_classes]
        logits = model(input_values)

        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred_tags = [label_list[i] for i in preds]

    segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration)

    if output_lab_path:
        os.makedirs(os.path.dirname(output_lab_path), exist_ok=True)
        save_lab(output_lab_path, segments_pred)
        print(f"Predictions saved to: {output_lab_path}")

    return segments_pred

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python infer.py <audio_path> <checkpoint_path> [<output_lab_path>] [<device>]")
        sys.exit(1)

    audio_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    config_path = sys.argv[3]
    output_lab_path = sys.argv[4] if len(sys.argv) > 3 else None
    device = sys.argv[5] if len(sys.argv) > 4 else "cuda"

    segments = infer_audio(
        audio_path=audio_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_lab_path=output_lab_path,
        device=device
    )

    print("Predicted segments:")
    for seg in segments:
        print(seg)
