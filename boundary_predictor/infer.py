import os
import torch
import torchaudio
import argparse
import yaml
import matplotlib.pyplot as plt
from model import BoundaryPredictor
from utils import pad_or_trim, median_cluster_filter
from preprocess import extract_mel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def infer(model, mel, threshold=0.5, min_separation_frames=1):
    model.eval()
    with torch.no_grad():
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(device)
        mel = mel.squeeze(1) if mel.dim() == 4 else mel
        out_main, _, _ = model(mel)
        pred = out_main.sigmoid().squeeze(0).cpu()

        indices = (pred > threshold).nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(indices, int):
            indices = [indices]

        filtered = median_cluster_filter(sorted(indices), min_distance=min_separation_frames)
        return filtered, pred

def plot(mel, pred, boundaries, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")

    for t in boundaries:
        ax.axvline(x=t, color="lightblue", linestyle="-", linewidth=1.2)

    ax.set_title("Predicted Boundaries")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def main(args):
    config = load_config(args.config)

    threshold = config.get("postprocess", {}).get("threshold", 0.5)
    min_sep_sec = config.get("postprocess", {}).get("min_separation_sec", 0.005)
    hop = config["data"]["hop_length"]
    sr = config["data"]["sample_rate"]
    min_separation_frames = max(1, int(min_sep_sec * sr / hop))

    model = BoundaryPredictor(
        mel_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"]
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded checkpoint from {args.checkpoint}")

    wav, sr_loaded = torchaudio.load(args.wav)
    wav = wav.mean(dim=0, keepdim=True)
    expected_sr = config["data"]["sample_rate"]
    if sr_loaded != expected_sr:
        print(f"Input audio is {sr_loaded}Hz, resampling to {expected_sr}Hz for boundary detection")
        resampler = torchaudio.transforms.Resample(orig_freq=sr_loaded, new_freq=expected_sr)
        wav = resampler(wav)
        sr_loaded = expected_sr

    mel = extract_mel(wav, sr_loaded)
    boundaries, pred = infer(model, mel, threshold=threshold, min_separation_frames=min_separation_frames)

    if args.plot:
        plot(mel[0], pred, boundaries, args.plot)

    if args.out_txt:
        times_sec = [f"{(frame * hop) / sr:.4f}" for frame in boundaries]
        with open(args.out_txt, "w") as f:
            f.write("\n".join(times_sec))
        print(f"Saved timestamps to {args.out_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--plot", type=str, default=None, help="Path to save visualization PNG")
    parser.add_argument("--out_txt", type=str, default=None, help="Path to save timestamps TXT")
    args = parser.parse_args()
    main(args)
