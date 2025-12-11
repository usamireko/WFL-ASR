import os
import json
import matplotlib.pyplot as plt
import numpy as np

def decode_bio_tags(tags, frame_duration=0.02, offsets=None):
    segments = []
    curr_ph, start_idx = None, None
    
    def finalize(end_idx):
        nonlocal curr_ph, start_idx
        s_time = (start_idx + (offsets[start_idx,0] if offsets is not None else 0.0)) * frame_duration
        e_time = (end_idx + (offsets[end_idx,1] if offsets is not None else 1.0)) * frame_duration
        segments.append((s_time, e_time, curr_ph))
        curr_ph, start_idx = None, None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if curr_ph: finalize(i)
            curr_ph = tag[2:]
            start_idx = i
        elif tag == "O":
            if curr_ph: finalize(i)
        elif tag.startswith("I-"):
            ph = tag[2:]
            if ph != curr_ph:
                if curr_ph: finalize(i)
                curr_ph = ph
                start_idx = i
                
    if curr_ph: finalize(len(tags))
    return segments

def save_lab(path, segments):
    with open(path, "w") as f:
        for s, e, p in segments:
            f.write(f"{int(s*1e7)} {int(e*1e7)} {p}\n")

def load_phoneme_list(path):
    with open(path, "r") as f: return [l.strip() for l in f if l.strip()]

def load_langs(path):
    d = {}
    with open(path, "r") as f:
        for l in f: k,v = l.strip().split(","); d[k] = int(v)
    return d

def load_phoneme_merge_map(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return None

def canonical_to_lang(ph, lang, m_map):
    return m_map.get(ph, {}).get(lang, ph)

def merge_adjacent_segments(segs, mode="right"):
    if not segs or mode=="none": return segs
    merged = [segs[0]]
    for s, e, p in segs[1:]:
        ls, le, lp = merged[-1]
        if p == lp: merged[-1] = (ls, e, lp)
        else: merged.append((s, e, p))
    return merged

def visualize_prediction(wav, sr, pred, gt=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(np.linspace(0, len(wav)/sr, len(wav)), wav, color="lightblue", alpha=0.6)
    
    for s, e, p in pred:
        ax.axvline(s, c="red", alpha=0.5, ls="--")
        if e-s > 0.02: ax.text((s+e)/2, 0.8, p, c="red", ha="center", fontsize=8)
            
    if gt:
        for item in gt:
            if len(item)==3:
                s, e, p = item
                ax.axvline(s, c="green", alpha=0.3)
                if e-s > 0.02: ax.text((s+e)/2, 0.6, p, c="green", ha="center", fontsize=8)
    return fig
