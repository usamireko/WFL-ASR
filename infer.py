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

def align_phoneme_list(segments_pred, forced_list):
    result = []
    pred_idx = 0
    forced_idx = 0
    used_preds = set()

    pred_map = [None] * len(forced_list)
    for f_i, f_ph in enumerate(forced_list):
        for p_i in range(pred_idx, len(segments_pred)):
            _, _, p_ph = segments_pred[p_i]
            if p_ph == f_ph and p_i not in used_preds:
                pred_map[f_i] = p_i
                used_preds.add(p_i)
                pred_idx = p_i + 1
                break
    pred_ptr = 0
    for f_i, f_ph in enumerate(forced_list):
        if pred_map[f_i] is None:
            while pred_ptr < len(segments_pred) and pred_ptr in used_preds:
                pred_ptr += 1
            if pred_ptr < len(segments_pred):
                pred_map[f_i] = pred_ptr
                used_preds.add(pred_ptr)
                pred_ptr += 1

    for f_i, f_ph in enumerate(forced_list):
        p_i = pred_map[f_i]
        if p_i is not None and p_i < len(segments_pred):
            s, e, _ = segments_pred[p_i]
            result.append((s, e, f_ph))
    return result

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

def infer_audio(audio_path, config_path="config.yaml", checkpoint_path="best_model.pt", output_lab_path=None, device="cuda", lang_id=None):
    config = load_config(config_path)
    phoneme_txt = audio_path.replace(".wav", ".txt")
    forced = None

    labels = load_phoneme_list(os.path.join(config["output"]["save_dir"], "phonemes.txt"))
    model = BIOPhonemeTagger(config, labels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()

    if os.path.exists(phoneme_txt):
        forced = []
        with open(phoneme_txt, "r", encoding="utf-8") as f:
            for line in f:
                forced.extend(line.strip().split())
        print(f"Loaded forced phoneme list with {len(forced)} phonemes.")

    audio, sr = sf.read(audio_path)
    if sr != config["data"]["sample_rate"]:
        audio = torchaudio.functional.resample(
            torch.tensor(audio), orig_freq=sr, new_freq=config["data"]["sample_rate"]
        ).numpy()
        sr = config["data"]["sample_rate"]

    if len(audio) > 0:
        audio = audio / (max(abs(audio)) + 1e-8)

    if len(audio) / sr > MAX_SEGMENT_DURATION:
        print(f"Audio is too long ({len(audio)/sr:.1f}s), splitting...")
        segments = split_audio(audio, sr)
        segments_pred = process_segments(model, segments, sr, config, device, lang_id)
    else:
        inp = torch.tensor(audio, dtype=torch.float32).to(device)
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

        logits_list = []

        if lang_id is not None:
            lt = torch.tensor([lang_id], dtype=torch.long).to(device)
            out = model(inp, lt)
            logits = out[0] if isinstance(out, tuple) else out
            logits_list.append(logits)
        else:
            for lid in lang2id.values():
                lt = torch.tensor([lid], dtype=torch.long).to(device)
                out = model(inp, lt)
                logits = out[0] if isinstance(out, tuple) else out
                logits_list.append(logits)

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        pred_ids = torch.argmax(avg_logits, dim=-1).squeeze(0).cpu().numpy()
        smoothed_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])
        tags = [labels[i] for i in smoothed_ids]
        segments_pred = decode_bio_tags(tags, frame_duration=frame_duration)

    if config["postprocess"]["merge_segments"] != "none":
        segments_pred = merge_adjacent_segments(segments_pred, mode=config["postprocess"]["merge_segments"])

    if forced is not None:
        aligned = align_phoneme_list(segments_pred, forced)
        if "SP" not in forced and "AP" not in forced:
            before = [s for s in segments_pred if s[2] in ("SP", "AP") and s[1] <= aligned[0][0]]
            after = [s for s in segments_pred if s[2] in ("SP", "AP") and s[0] >= aligned[-1][1]]
            segments_pred = before + aligned + after
        else:
            segments_pred = aligned

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
