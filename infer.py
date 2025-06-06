import sys
import os
import torch
import yaml
import soundfile as sf
import torchaudio
import numpy as np
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list, merge_adjacent_segments, load_langs
from scipy.ndimage import median_filter

frame_duration = 0.02  # ~20ms per frame
MAX_SEGMENT_DURATION = 30.0

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def split_audio(audio, sr, max_duration=MAX_SEGMENT_DURATION):
    total_samples = len(audio)
    samples_per_segment = int(max_duration * sr)
    segments = []
    
    for start in range(0, total_samples, samples_per_segment):
        end = min(start + samples_per_segment, total_samples)
        segments.append(audio[start:end])
    
    return segments

def process_segments(model, segments, sr, config, device, lang_id=None):
    all_segments = []
    current_time = 0.0
    
    for segment in segments:
        if len(segment) > 0:
            segment = segment / (max(abs(segment)) + 1e-8)
        
        input_values = torch.tensor(segment, dtype=torch.float32).to(device)
        
        logits_list = []
        encoder_outs = []
        
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))
        
        if lang_id is not None:
            lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
            output = model(input_values, lang_tensor)
            logits, encoder_out = output if isinstance(output, tuple) else (output, None)
            logits_list.append(logits)
            encoder_outs.append(encoder_out)
        else:
            for lid in lang2id.values():
                lang_tensor = torch.tensor([lid], dtype=torch.long).to(device)
                output = model(input_values, lang_tensor)
                logits, encoder_out = output if isinstance(output, tuple) else (output, None)
                logits_list.append(logits)
                encoder_outs.append(encoder_out)
        
        stacked_logits = torch.stack(logits_list)
        avg_logits = torch.mean(stacked_logits, dim=0)
        
        pred_ids = torch.argmax(avg_logits, dim=-1).squeeze(0).cpu().numpy()
        smoothed_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])
        
        pred_tags = [model.id2label[i] for i in smoothed_ids]
        segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration)
        
        shifted_segments = []
        for start, end, ph in segments_pred:
            shifted_segments.append((start + current_time, end + current_time, ph))
        
        all_segments.extend(shifted_segments)
        current_time += len(segment) / sr
    
    return all_segments

def infer_folder(
    folder_path: str,
    config_path: str = "config.yaml",
    checkpoint_path: str = "best_model.pt",
    output_dir: str = "outputs",
    device: str = "cuda",
    lang_id: int = None
):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    os.makedirs(output_dir, exist_ok=True)

    for wav_file in wav_files:
        full_audio_path = os.path.join(folder_path, wav_file)
        output_lab_path = os.path.join(output_dir, wav_file.replace(".wav", ".lab"))

        print(f"\nInferencing: {wav_file}")
        segments = infer_audio(
            audio_path=full_audio_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_lab_path=output_lab_path,
            device=device,
            lang_id=lang_id
        )

        print("Predicted segments:")
        for seg in segments:
            start, end, ph = seg
            print(f"({round(start, 2)}, {round(end, 2)}, {ph})")

def infer_audio(
    audio_path: str,
    config_path: str = "config.yaml",
    checkpoint_path: str = "best_model.pt",
    output_lab_path: str = None,
    device: str = "cuda",
    lang_id: int = None
):
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

    if len(audio) / sr > MAX_SEGMENT_DURATION:
        print(f"Audio is too long ({len(audio)/sr:.2f}s), splitting into segments...")
        segments = split_audio(audio, sr)
        segments_pred = process_segments(model, segments, sr, config, device, lang_id)
    else:
        if len(audio) > 0:
            audio = audio / (max(abs(audio)) + 1e-8)
        
        input_values = torch.tensor(audio, dtype=torch.float32).to(device)
        
        logits_list = []
        encoder_outs = []
        
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))
        
        if lang_id is not None:
            lang_name = next((k for k, v in lang2id.items() if v == lang_id), f"unknown_id_{lang_id}")
            print(f"Inferencing with lang_id {lang_id} ({lang_name})")
            lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
            output = model(input_values, lang_tensor)
            logits, encoder_out = output if isinstance(output, tuple) else (output, None)
            logits_list.append(logits)
            encoder_outs.append(encoder_out)
        else:
            for lang_name, lid in lang2id.items():
                print(f"Inferencing with lang_id {lid} ({lang_name})")
                lang_tensor = torch.tensor([lid], dtype=torch.long).to(device)
                output = model(input_values, lang_tensor)
                logits, encoder_out = output if isinstance(output, tuple) else (output, None)
                logits_list.append(logits)
                encoder_outs.append(encoder_out)
        
        stacked_logits = torch.stack(logits_list)
        avg_logits = torch.mean(stacked_logits, dim=0)
        
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
        print("Folder     : python infer.py --folder <folder_path> <checkpoint_path> <config_path> [<output_dir>] [<device>] [--lang_id <id>]")
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
            start, end, ph = seg
            print(f"({round(start, 2)}, {round(end, 2)}, {ph})")
