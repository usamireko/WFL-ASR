import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

htk_time_factor = 1e7  # htk label uses 100ns units

def decode_bio_tags(tags, frame_duration=0.02):
    # bio2htk (start_time, end_time, phoneme)
    segments = []
    current_ph = None
    start_idx = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if current_ph is not None:
                end_time = i * frame_duration
                segments.append((start_idx * frame_duration, end_time, current_ph))
                current_ph = None
                start_idx = None
            continue

        if tag.startswith("B-"):
            if current_ph is not None:
                end_time = i * frame_duration
                segments.append((start_idx * frame_duration, end_time, current_ph))

            current_ph = tag[2:]
            start_idx = i

        elif tag.startswith("I-"):
            ph = tag[2:]
            if current_ph != ph:
                # mismatched I- tag without B-, treat it as B-
                if current_ph is not None:
                    end_time = i * frame_duration
                    segments.append((start_idx * frame_duration, end_time, current_ph))
                current_ph = ph
                start_idx = i

    if current_ph is not None:
        end_time = len(tags) * frame_duration
        segments.append((start_idx * frame_duration, end_time, current_ph))

    return segments

def save_lab(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for start, end, ph in segments:
            start_htk = int(start * htk_time_factor)
            end_htk = int(end * htk_time_factor)
            f.write(f"{start_htk} {end_htk} {ph}\n")

def load_phoneme_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def clean_label(ph):
    # for tensorboard but mf doesnt work im sobbing
    if isinstance(ph, list):
        ph = " ".join(str(x) for x in ph)

    ph = str(ph).strip()

    if ph.startswith("(") and ph.endswith(")"):
        ph = ph[1:-1].strip()

    if (ph.startswith("'") and ph.endswith("'")) or (ph.startswith('"') and ph.endswith('"')):
        ph = ph[1:-1].strip()

    return ph

def visualize_prediction(waveform, sample_rate, segments_pred, segments_gt=None, title="Prediction"):
    while isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
        segments_gt = segments_gt[0]

    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, len(waveform))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time, waveform, alpha=0.6, label="Waveform")

    # Pred red
    for (start, end, ph) in segments_pred:
        ph = clean_label(ph)
        ax.axvspan(start, end, color="red", alpha=0.2)
        ax.text((start + end) / 2, 0.9, ph, color="red", ha="center", transform=ax.get_xaxis_transform(), fontsize=12)

    # GT greem
    if segments_gt:
        for item in segments_gt:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                print(f"[WARN] Skipping malformed GT segment: {item}")
                continue
            start, end, ph = item
            try:
                start = float(start)
                end = float(end)
                ph = clean_label(ph)
                ax.axvspan(start, end, color="green", alpha=0.2)
                ax.text((start + end) / 2, 0.7, ph, color="green", ha="center", transform=ax.get_xaxis_transform(), fontsize=12)
            except Exception as e:
                print(f"[ERROR] Failed to plot GT segment {item}: {e}") # training explodes with invalid segment

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1, 1)
    return fig
