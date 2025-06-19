import os
import glob
import soundfile as sf
from tqdm import tqdm
import yaml
import json

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_lab(lab_path):
    phonemes = []
    with open(lab_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()

            if len(parts) != 3:
                print(f"[WARN] Skipping malformed line {line_num} in {lab_path}: {line.strip()}")
                continue

            try:
                start = int(parts[0]) / 1e7
                end = int(parts[1]) / 1e7
                ph = parts[2]
                phonemes.append((start, end, ph))
            except Exception as e:
                print(f"[ERROR] Failed to parse line {line_num} in {lab_path}: {e}")
                continue

    return phonemes

def to_bio_tags(phonemes, num_frames, frame_duration):
    tags = ["O"] * num_frames
    for start, end, ph in phonemes:
        start_idx = int(start / frame_duration)
        end_idx = int(end / frame_duration)
        if end_idx >= num_frames:
            end_idx = num_frames - 1
        if start_idx >= num_frames:
            continue
        tags[start_idx] = f"B-{ph}"
        for i in range(start_idx + 1, end_idx + 1):
            if i < num_frames:
                tags[i] = f"I-{ph}"
    return tags

def build_merge_map(groups):
    merge_map = {}
    reverse_map = {}
    for group in groups or []:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            continue
        canonical = group[0]
        if "/" not in canonical:
            continue
        _, canonical_ph = canonical.split("/", 1)
        for item in group:
            if "/" not in item:
                continue
            lang, ph = item.split("/", 1)
            merge_map.setdefault(lang, {})[ph] = canonical_ph
            reverse_map.setdefault(canonical_ph, {})[lang] = ph
    return merge_map, reverse_map
    
def preprocess(data_dir, config):
    frame_duration = config["data"].get("frame_duration", 0.02)
    all_lang_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    merge_map, reverse_map = build_merge_map(config.get("merged_phoneme_groups", []))
    save_dir = config["output"]["save_dir"]
    existing_lang2id = {}
    existing_phonemes = set()

    langs_txt_path = os.path.join(save_dir, "langs.txt")
    phonemes_txt_path = os.path.join(save_dir, "phonemes.txt")

    if os.path.exists(langs_txt_path):
        with open(langs_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    lang, idx = parts
                    existing_lang2id[lang] = int(idx)

    if os.path.exists(phonemes_txt_path):
        with open(phonemes_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line != "O":
                    if line.startswith("B-") or line.startswith("I-"):
                        existing_phonemes.add(line[2:])

    lang2id = dict(existing_lang2id)
    next_lang_id = max(lang2id.values(), default=-1) + 1
    for lang in all_lang_dirs:
        if lang not in lang2id:
            lang2id[lang] = next_lang_id
            next_lang_id += 1

    dataset = []
    phoneme_set = set()
    lang_phonemes = {}

    for lang in all_lang_dirs:
        lang_path = os.path.join(data_dir, lang)
        wav_files = sorted(glob.glob(os.path.join(lang_path, "*.wav")))
        lang_phonemes[lang] = set()

        for wav_path in tqdm(wav_files, desc=f"[{lang}]"):
            base = os.path.splitext(os.path.basename(wav_path))[0]
            lab_path = os.path.join(lang_path, base + ".lab")

            if not os.path.exists(lab_path):
                print(f"Missing label for {base}, skipping.")
                continue

            audio, sr = sf.read(wav_path)
            duration = len(audio) / sr
            num_frames = int(duration / frame_duration)

            phoneme_segments_raw = parse_lab(lab_path)
            phoneme_segments = []
            for start, end, ph in phoneme_segments_raw:
                merged_ph = merge_map.get(lang, {}).get(ph, ph)
                phoneme_segments.append((start, end, merged_ph))
                phoneme_set.add(merged_ph)
                lang_phonemes[lang].add(merged_ph)

            bio_tags = to_bio_tags(phoneme_segments, num_frames, frame_duration)

            sample = {
                "wav_path": wav_path,
                "bio_tags": bio_tags,
                "phoneme_segments": phoneme_segments,
                "lang_id": lang2id[lang]
            }
            dataset.append(sample)

    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    dataset_json_path = os.path.join(save_dir, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(dataset, f, indent=2)

    if reverse_map:
        merge_map_path = os.path.join(save_dir, "phoneme_merge_map.json")
        with open(merge_map_path, "w", encoding="utf-8") as f:
            json.dump(reverse_map, f, indent=2, ensure_ascii=False)
        
    merged_phonemes = set(existing_phonemes).union(phoneme_set)
    all_tags = {f"B-{ph}" for ph in merged_phonemes}
    all_tags.update({f"I-{ph}" for ph in merged_phonemes})
    all_tags.add("O")

    phoneme_txt_path = os.path.join(save_dir, "phonemes.txt")
    with open(phoneme_txt_path, "w", encoding="utf-8") as f:
        for tag in sorted(all_tags):
            f.write(f"{tag}\n")

    langs_txt_path = os.path.join(save_dir, "langs.txt")
    with open(langs_txt_path, "w", encoding="utf-8") as f:
        for lang, idx in lang2id.items():
            f.write(f"{lang},{idx}\n")

    if merge_map:
        print("\nApplied merged phoneme groups:")
        for lang, mapping in merge_map.items():
            for src, tgt in mapping.items():
                print(f"  {lang}/{src} -> {tgt}")

    print(f"\nProcessed {len(dataset)} samples.")
    print(f"\nGenerated {len(all_tags)} BIO labels -> {phoneme_txt_path}")
    print(f"\nSaved language mapping -> {langs_txt_path}")
    print(f"\nSaved language phoneme list -> {lang_phonemes_json_path}")
    if reverse_map:
        print(f"\nSaved phoneme merge map -> {merge_map_path}")

    print("\nPhoneme usage by language:")
    for lang, phonemes in lang_phonemes.items():
        sorted_phs = sorted(list(phonemes))
        print(f"  {lang}: {sorted_phs}")

    config["model"]["num_languages"] = len(lang2id)
    updated_config_path = os.path.join(save_dir, "config.yaml")
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"\nSaved updated config -> {updated_config_path}")

if __name__ == "__main__":
    config = load_config()
    preprocess(config["data"]["data_dir"], config)
