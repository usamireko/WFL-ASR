import os
import glob
import yaml
import json
import soundfile as sf
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

htk_time_factor = 1e7  # 100ns
frame_hop = 256
n_fft = 1024
n_mels = 80

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_lab(lab_path):
    ends = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, end_htk, _ = parts
            end = int(end_htk) / htk_time_factor
            ends.append(end)
    return sorted(set(ends))

def extract_mel(wav, sr):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=frame_hop,
        n_mels=n_mels,
        power=2.0,
        normalized=True
    )(wav)
    return mel_spec.log2().clamp(min=-11.52)

def build_soft_spike_map(boundaries_sec, num_frames, hop_length, sr, sigma=0.015):
    times = np.arange(num_frames) * hop_length / sr
    spike_map = np.zeros_like(times)
    for t in boundaries_sec:
        spike_map += np.exp(-0.5 * ((times - t) / sigma) ** 2)
    spike_map = np.clip(spike_map, 0, 1.0)
    return spike_map.tolist()

def preprocess_boundary_dataset(data_dir, config):
    save_dir = config["output"]["save_dir"]
    data_save_dir = os.path.join(save_dir, "boundary_data")
    os.makedirs(data_save_dir, exist_ok=True)

    all_samples = []
    all_wav_files = sorted(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))

    for wav_path in tqdm(all_wav_files, desc="Processing"):
        base = os.path.splitext(os.path.basename(wav_path))[0]
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        if not os.path.exists(lab_path):
            continue

        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)

        mel = extract_mel(wav, sr)
        mel_len = mel.shape[-1]

        boundaries = load_lab(lab_path)
        spike_map = build_soft_spike_map(boundaries, mel_len, frame_hop, sr, sigma=config["data"].get("sigma", 0.015))

        mel_path = os.path.join(data_save_dir, base + ".mel.pt")
        spike_path = os.path.join(data_save_dir, base + ".spike.pt")
        torch.save(mel, mel_path)
        torch.save(torch.tensor(spike_map), spike_path)

        sample = {
            "mel": mel_path,
            "spike": spike_path
        }
        all_samples.append(sample)

    dataset_json_path = os.path.join(save_dir, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    config_path = os.path.join(save_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Saved {len(all_samples)} -> {data_save_dir}")
    print(f"Saved dataset.json -> {dataset_json_path}")
    print(f"Saved config.yaml -> {config_path}")

if __name__ == "__main__":
    config = load_config("config.yaml")
    preprocess_boundary_dataset(config["data"]["data_dir"], config)
