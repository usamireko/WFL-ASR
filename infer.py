import sys
import os
import torch
import yaml
import soundfile as sf
import torchaudio
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list, merge_adjacent_segments, load_langs
from scipy.ndimage import median_filter

frame_duration = 0.02  # ~20ms per frame

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def infer_folder(
    folder_path: str,
    config_path: str = "config.yaml",
    checkpoint_path: str = "best_model.pt",
    output_dir: str = "outputs",
    device: str = "cuda"
):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    os.makedirs(output_dir, exist_ok=True)

    for wav_file in wav_files:
        full_audio_path = os.path.join(folder_path, wav_file)
        output_lab_path = os.path.join(output_dir, wav_file.replace(".wav", ".lab"))

        print(f"Inferencing: {wav_file}")
        segments = infer_audio(
            audio_path=full_audio_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_lab_path=output_lab_path,
            device=device
        )

        for seg in segments:
            print(seg)

def infer_audio(
    audio_path: str,
    config_path: str = "config.yaml",
    checkpoint_path: str = "best_model.pt",
    output_lab_path: str = None,
    device: str = "cuda",
    lang_id: int = None
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
    
    label_list = load_phoneme_list(os.path.join(config["output"]["save_dir"], "phonemes.txt"))
    median_filter_size = config["postprocess"]["median_filter"]
    merge_segments = config["postprocess"]["merge_segments"]

    model = BIOPhonemeTagger(config, label_list)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
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

    logits_list = []

    lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

    if lang_id is not None:
        lang_name = next((k for k, v in lang2id.items() if v == lang_id), f"unknown_id_{lang_id}")
        print(f"Inferencing with lang_id {lang_id} ({lang_name})")

        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
        logits = model(input_values, lang_tensor)
        logits_list.append(logits)
    else:
        for lang_name, lid in lang2id.items():
            print(f"Inferencing with lang_id {lid} ({lang_name})")
            lang_tensor = torch.tensor([lid], dtype=torch.long).to(device)
            logits = model(input_values, lang_tensor)
            logits_list.append(logits)

    stacked_logits = torch.stack(logits_list)  # [N_langs, 1, T, C]
    avg_logits = torch.mean(stacked_logits, dim=0)  # [1, T, C]

    pred_ids = torch.argmax(avg_logits, dim=-1).squeeze(0).cpu().numpy()
    smoothed_ids = median_filter(pred_ids, size=median_filter_size)

    pred_tags = [label_list[i] for i in smoothed_ids]
    segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration)

    if merge_segments != "none":
        segments_pred = merge_adjacent_segments(segments_pred, mode=merge_segments)

    if output_lab_path:
        dir_path = os.path.dirname(output_lab_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        save_lab(output_lab_path, segments_pred)
        print(f"Predictions saved to: {output_lab_path}")

    return segments_pred

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage:")
        print("Single file: python infer.py <audio_path> <checkpoint_path> <config_path> [<output_lab_path>] [<device>]")
        print("Folder     : python infer.py --folder <folder_path> <checkpoint_path> <config_path> [<output_dir>] [<device>]")
        sys.exit(1)

    lang_id = None
    if "--lang_id" in sys.argv:
        lang_idx = sys.argv.index("--lang_id")
        lang_id = int(sys.argv[lang_idx + 1])
        sys.argv.pop(lang_idx + 1)
        sys.argv.pop(lang_idx)

    if sys.argv[1] == "--folder":
        folder_path = sys.argv[2]
        checkpoint_path = sys.argv[3]
        config_path = sys.argv[4]
        output_dir = sys.argv[5] if len(sys.argv) > 5 else "outputs"
        device = sys.argv[6] if len(sys.argv) > 6 else "cuda"

        infer_folder(
            folder_path=folder_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=device,
            lang_id=lang_id
        )
    else:
        audio_path = sys.argv[1]
        checkpoint_path = sys.argv[2]
        config_path = sys.argv[3]
        output_lab_path = sys.argv[4] if len(sys.argv) > 4 else None
        device = sys.argv[5] if len(sys.argv) > 5 else "cuda"

        segments = infer_audio(
            audio_path=audio_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_lab_path=output_lab_path,
            device=device,
            lang_id=lang_id
        )

        print("Predicted segments:")
        for seg in segments:
            print(seg)
