import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

htk_time_factor = 1e7  # htk label uses 100ns units

def decode_bio_tags(tags, frame_duration=0.02, offsets=None):
    # bio2htk (start_time, end_time, phoneme)
    segments = []
    current_ph = None
    start_idx = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if current_ph is not None:
                end_idx = i
                start_time = (start_idx + 0.5) * frame_duration
                end_time = (end_idx + 0.5) * frame_duration

                if offsets is not None:
                    start_time = (start_idx + offsets[start_idx][0].item()) * frame_duration
                    end_time = (end_idx + offsets[end_idx][1].item()) * frame_duration

                segments.append((start_time, end_time, current_ph))
                current_ph = None
                start_idx = None
            continue

        if tag.startswith("B-"):
            if current_ph is not None:
                end_idx = i
                start_time = (start_idx + 0.5) * frame_duration
                end_time = (end_idx + 0.5) * frame_duration

                if offsets is not None:
                    start_time = (start_idx + offsets[start_idx][0].item()) * frame_duration
                    end_time = (end_idx + offsets[end_idx][1].item()) * frame_duration

                segments.append((start_time, end_time, current_ph))

            current_ph = tag[2:]
            start_idx = i

        elif tag.startswith("I-"):
            ph = tag[2:]
            if current_ph != ph:
                if current_ph is not None:
                    end_idx = i
                    start_time = (start_idx + 0.5) * frame_duration
                    end_time = (end_idx + 0.5) * frame_duration

                    if offsets is not None:
                        start_time = (start_idx + offsets[start_idx][0].item()) * frame_duration
                        end_time = (end_idx + offsets[end_idx][1].item()) * frame_duration

                    segments.append((start_time, end_time, current_ph))
                current_ph = ph
                start_idx = i

    if current_ph is not None:
        end_idx = len(tags) - 1
        start_time = (start_idx + 0.5) * frame_duration
        end_time = (end_idx + 0.5) * frame_duration

        if offsets is not None and start_idx < len(offsets) and end_idx < len(offsets):
            start_time = (start_idx + offsets[start_idx][0].item()) * frame_duration
            end_time = (end_idx + offsets[end_idx][1].item()) * frame_duration

        segments.append((start_time, end_time, current_ph))

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
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(time, waveform, alpha=0.8, color="lightblue", zorder=0)

    for start, end, ph in segments_pred:
        ph = clean_label(ph)
        text_pos = (start + end) / 2
        if end - start > 0.02:
            ax.text(text_pos, 0.9, ph, color="red", ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(), fontsize=12, zorder=3)
        ax.axvline(start, color="red", linestyle="-", linewidth=0.6, alpha=0.5, zorder=2)

    if segments_gt:
        for item in segments_gt:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            try:
                start, end, ph = float(item[0]), float(item[1]), clean_label(item[2])
                text_pos = (start + end) / 2
                if end - start > 0.02:
                    ax.text(text_pos, 0.7, ph, color="green", ha="center", va="bottom",
                            transform=ax.get_xaxis_transform(), fontsize=12, zorder=3)
                ax.axvline(start, color="green", linestyle="-", linewidth=0.6, alpha=0.5, zorder=2)
            except Exception as e:
                print(f"[ERROR] Failed to plot GT segment {item}: {e}")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-1, 1)

    legend_labels = [
        plt.Line2D([], [], linestyle="none", marker='o', color="red", markersize=8, label="Pred"),
        plt.Line2D([], [], linestyle="none", marker='o', color="green", markersize=8, label="GT"),
    ]
    ax.legend(handles=legend_labels, loc="upper right", frameon=True, fancybox=True, framealpha=0.6)

    return fig

def merge_adjacent_segments(segments, mode="right"):
    if not segments or mode == "none":
        return segments

    merged = []

    if mode == "right":
        merged = [segments[0]]
        for start, end, ph in segments[1:]:
            last_start, last_end, last_ph = merged[-1]
            if ph == last_ph:
                merged[-1] = (last_start, end, ph)
            else:
                merged.append((start, end, ph))
    elif mode == "left":
        i = 0
        while i < len(segments):
            if i > 0 and segments[i][2] == segments[i - 1][2]:
                prev_start, prev_end, ph = merged.pop()
                merged.append((prev_start, segments[i][1], ph))
            else:
                merged.append(segments[i])
            i += 1
    elif mode == "previous":
        i = 0
        while i < len(segments):
            if i > 1 and segments[i - 1][2] == segments[i][2]:
                if len(merged) >= 2:
                    p0 = merged[-2]  # previous previous
                    p1 = merged.pop()  # previous
                    merged[-1] = (p0[0], segments[i][1], p0[2])
                else:
                    merged.append(segments[i])
            else:
                merged.append(segments[i])
            i += 1
    else:
        raise ValueError(f"Unsupported merge mode: {mode}")
    return merged

def load_langs(lang_path):
    lang2id = {}
    with open(lang_path, "r", encoding="utf-8") as f:
        for line in f:
            lang, idx = line.strip().split(",")
            lang2id[lang] = int(idx)
    return lang2id
