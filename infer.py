import sys
import os
import torch
import yaml
import soundfile as sf
import torchaudio
import torch
import numpy as np
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list, merge_adjacent_segments, load_langs, load_phoneme_merge_map, canonical_to_lang 
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

def sample_from_logits(logits, k=5, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
    return topk_indices.gather(1, sampled.unsqueeze(-1)).squeeze(-1)

def top_p_sample(logits, p=0.9, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs <= p
    mask[..., 0] = True

    filtered_probs = torch.zeros_like(probs)
    for t in range(probs.size(0)):
        valid_idx = sorted_indices[t][mask[t]]
        filtered_probs[t][valid_idx] = probs[t][valid_idx]
        filtered_probs[t] /= filtered_probs[t].sum()

    sampled = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    return sampled

def suppress_low_confidence(logits, id2label, threshold=0.5):
    import torch
    probs = torch.softmax(logits, dim=-1)
    max_probs, pred_ids = torch.max(probs, dim=-1)
    smoothed_tags = []
    for prob, idx in zip(max_probs, pred_ids):
        if prob < threshold:
            smoothed_tags.append("O")
        else:
            smoothed_tags.append(id2label[idx.item()])
    return smoothed_tags

def process_segments(model, segments, sr, config, device, lang_id=None,
                     sample=False, top_k=0, top_p=0.0, temperature=1.0,
                     cache_dir=None, base_name=None, confidence_threshold=0.0,
                     merge_map=None):
    all_segments = []
    current_time = 0.0

    lang_name = None
    if lang_id is not None:
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))
        for n, i in lang2id.items():
            if i == lang_id:
                lang_name = n
                break

    for idx, segment in enumerate(segments):
        if len(segment) > 0:
            segment = segment / (max(abs(segment)) + 1e-8)

        seg_logits = None
        seg_offsets = None

        use_cache = cache_dir is not None and base_name is not None
        if use_cache:
            seg_logit_path = os.path.join(cache_dir, f"{base_name}_seg{idx}_logits.pt")
            seg_offset_path = os.path.join(cache_dir, f"{base_name}_seg{idx}_offsets.pt")
            if os.path.exists(seg_logit_path):
                print(f"Loaded cached logits for segment {idx}")
                seg_logits = torch.load(seg_logit_path, map_location=device, weights_only=False)
                if os.path.exists(seg_offset_path):
                    seg_offsets = torch.load(seg_offset_path, map_location=device, weights_only=False)

        if seg_logits is None:
            input_values = torch.tensor(segment, dtype=torch.float32).to(device)

            logits_list = []
            offsets_list = []
            lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

            if lang_id is not None:
                if lang_id > max(lang2id.values()):
                    raise ValueError(f"Language ID {lang_id} is invalid. Available: {lang2id}")
                lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
                output = model(input_values, lang_tensor)
                logits, offsets = output if isinstance(output, tuple) else (output, None)
                logits_list.append(logits)
                if offsets is not None:
                    offsets_list.append(offsets)
            else:
                for lid in lang2id.values():
                    lang_tensor = torch.tensor([lid], dtype=torch.long).to(device)
                    output = model(input_values, lang_tensor)
                    logits, offsets = output if isinstance(output, tuple) else (output, None)
                    logits_list.append(logits)
                    if offsets is not None:
                        offsets_list.append(offsets)

            seg_logits = torch.mean(torch.stack(logits_list), dim=0)
            seg_offsets = torch.mean(torch.stack(offsets_list), dim=0).squeeze(0) if offsets_list else None

            if use_cache:
                torch.save(seg_logits, seg_logit_path)
                if seg_offsets is not None:
                    torch.save(seg_offsets, seg_offset_path)

        logits_cpu = seg_logits.squeeze(0).cpu()
        pred_tags = suppress_low_confidence(
            logits_cpu, model.id2label,
            threshold=confidence_threshold
        )

        pred_ids = [model.label2id.get(tag, model.label2id["O"]) for tag in pred_tags]
        if config["postprocess"]["median_filter"] > 1:
            pred_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])
        pred_tags = [model.id2label[i] for i in pred_ids]

        segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=seg_offsets)
        if merge_map and lang_name:
            segments_pred = [
                (s, e, canonical_to_lang(ph, lang_name, merge_map))
                for s, e, ph in segments_pred
            ]
        shifted_segments = [(start + current_time, end + current_time, ph) for start, end, ph in segments_pred]
        all_segments.extend(shifted_segments)
        current_time += len(segment) / sr

    return all_segments

def infer_audio(audio_path, config_path="config.yaml", checkpoint_path="best_model.pt",
                output_lab_path=None, device="cuda", lang_id=None,
                sample=False, top_k=0, top_p=0.0, temperature=1.0,
                confidence_threshold=0.0):
    config = load_config(config_path)
    merge_map_path = os.path.join(config["output"]["save_dir"], "phoneme_merge_map.json")
    merge_map = load_phoneme_merge_map(merge_map_path) if os.path.exists(merge_map_path) else None
    phoneme_txt = audio_path.replace(".wav", ".txt")
    forced = None

    lang_name = None
    if lang_id is not None:
        lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))
        for n, i in lang2id.items():
            if i == lang_id:
                lang_name = n
                break

    labels = load_phoneme_list(os.path.join(config["output"]["save_dir"], "phonemes.txt"))
    model = BIOPhonemeTagger(config, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=sr, new_freq=config["data"]["sample_rate"]).numpy()
        sr = config["data"]["sample_rate"]

    #cache for re-inference
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    audio_dir = os.path.dirname(audio_path)
    cache_dir = os.path.join(audio_dir, ".wfl_cache")
    os.makedirs(cache_dir, exist_ok=True)
    logits_cache = os.path.join(cache_dir, f"{base_name}_logits.pt")
    offsets_cache = os.path.join(cache_dir, f"{base_name}_offsets.pt")

    avg_logits = None
    avg_offsets = None

    if len(audio) > 0:
        audio = audio / (max(abs(audio)) + 1e-8)

    if len(audio) / sr > MAX_SEGMENT_DURATION:
        print(f"Audio is too long ({len(audio)/sr:.1f}s), splitting...")
        segments = split_audio(audio, sr)
        segments_pred = process_segments(
            model, segments, sr, config, device, lang_id,
            sample=sample, top_k=top_k, top_p=top_p, temperature=temperature,
            cache_dir=cache_dir, base_name=base_name, confidence_threshold=confidence_threshold,
            merge_map=merge_map)
    else:
        if os.path.exists(logits_cache):
            print(f"Loaded cached logits for {base_name}")
            avg_logits = torch.load(logits_cache, map_location=device, weights_only=False)
            avg_offsets = torch.load(offsets_cache, map_location=device, weights_only=False) if os.path.exists(offsets_cache) else None
        else:
            inp = torch.tensor(audio, dtype=torch.float32).to(device)
            lang2id = load_langs(os.path.join(config["output"]["save_dir"], "langs.txt"))

            logits_list = []
            offsets_list = []

            if lang_id is not None:
                if lang_id > max(lang2id.values()):
                    raise ValueError(f"Error: Language ID ({lang_id}) is higher than the latest ID ({max(lang2id.values())}) of this model.\n Languages and Codes available: {lang2id}")
                lt = torch.tensor([lang_id], dtype=torch.long).to(device)
                out = model(inp, lt)
                logits, offsets = out if isinstance(out, tuple) else (out, None)
                logits_list.append(logits)
                if offsets is not None:
                    offsets_list.append(offsets)
            else:
                for lid in lang2id.values():
                    lt = torch.tensor([lid], dtype=torch.long).to(device)
                    out = model(inp, lt)
                    logits, offsets = out if isinstance(out, tuple) else (out, None)
                    logits_list.append(logits)
                    if offsets is not None:
                        offsets_list.append(offsets)

            avg_logits = torch.mean(torch.stack(logits_list), dim=0)
            avg_offsets = torch.mean(torch.stack(offsets_list), dim=0).squeeze(0) if offsets_list else None

            torch.save(avg_logits, logits_cache)
            if avg_offsets is not None:
                torch.save(avg_offsets, offsets_cache)

        logits_cpu = avg_logits.squeeze(0).cpu()
        if sample:
            if top_p > 0.0:
                pred_ids = top_p_sample(logits_cpu, p=top_p, temperature=temperature).numpy()
            elif top_k > 0:
                pred_ids = sample_from_logits(logits_cpu, k=top_k, temperature=temperature).numpy()
            else:
                pred_ids = torch.argmax(logits_cpu, dim=-1).numpy()
        else:
            pred_ids = torch.argmax(logits_cpu, dim=-1).numpy()
            
        pred_tags = suppress_low_confidence(
            logits_cpu, model.id2label,
            threshold=confidence_threshold
        )
        pred_ids = [model.label2id.get(tag, model.label2id["O"]) for tag in pred_tags]
        if config["postprocess"]["median_filter"] > 1:
            pred_ids = median_filter(pred_ids, size=config["postprocess"]["median_filter"])
        pred_tags = [model.id2label[i] for i in pred_ids]

        segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=avg_offsets)
        if merge_map and lang_name:
            segments_pred = [
                (s, e, canonical_to_lang(ph, lang_name, merge_map))
                for s, e, ph in segments_pred
            ]

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

def infer_folder(folder_path: str, config_path: str = "config.yaml", checkpoint_path: str = "best_model.pt",
                 output_dir: str = "outputs", device: str = "cuda", lang_id: int = None,
                 sample=False, top_k=0, top_p=0.0, temperature=1.0, confidence_threshold=0.0):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    os.makedirs(output_dir, exist_ok=True)

    for wav_file in wav_files:
        full_audio_path = os.path.join(folder_path, wav_file)
        output_lab_path = os.path.join(output_dir, wav_file.replace(".wav", ".lab"))

        print(f"\nInferencing: {wav_file}")
        segments = infer_audio(
            audio_path=str(full_audio_path),
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            output_lab_path=str(output_lab_path),
            device=device,
            lang_id=lang_id,
            sample=sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            confidence_threshold=confidence_threshold
        )
        print("Predicted segments:")
        for seg in segments:
            start, end, ph = seg
            print(f"({round(start, 2)}, {round(end, 2)}, {ph})")

if __name__ == "__main__":
    import click
    from pathlib import Path
    @click.command(help='Infer with WFL')
    @click.argument('path', metavar='PATH')
    @click.option('--checkpoint', '-ckpt', type=str, required=True, help='Path to WFL Checkpoint.')
    @click.option('--config', '-c', type=str, required=True, help='Path to Config file.')
    @click.option('--output', '-o', type=str, required=False, default=".", help='Path to output labels.')
    @click.option('--lang-id', '-l', type=int, required=False, default=None, help='Language ID.')
    @click.option('--sample', '-s', is_flag=True, help='Enable sampling instead of argmax')
    @click.option('--top-k', '-tk', type=int, default=0, help='Top-K sampling (range: 1-20)')
    @click.option('--top-p', '-tp', type=float, default=0.0, help='Top-P sampling (range: 0.1-1)')
    @click.option('--temperature', '-temp', type=float, default=1.0, help='Sampling temperature (range: 0.1-2)')
    @click.option('--device', '-d', type=str, default="auto", help='Device to use: "cuda", "cuda:0", or "cpu". Auto-detects if not specified.')
    @click.option('--confidence-threshold', '-ct', type=float, default=None, help='Suppress predictions with low confidence. Set 0 to disable.')

    def main(path, checkpoint, config, output, lang_id, sample, top_k, top_p, temperature, device, confidence_threshold):
        # I feel like a yandere sim dev doing this
        if sample:
            if top_k <= 0 and top_p <= 0.0:
                print("Sampling is enabled but neither --top-k nor --top-p is set.")
                sys.exit(1)
            if top_k > 0 and top_p > 0.0:
                print("You can't use both --top-k and --top-p at the same time.")
                sys.exit(1)
            if top_k < 0:
                print("top-k must be ≥ 1.")
                sys.exit(1)
            if top_p < 0.0 or top_p > 1.0:
                print("top-p must be between 0.1 and 1.0.")
                sys.exit(1)
            if temperature <= 0.0:
                print("temperature must be greater than 0.")
                sys.exit(1)

        requested_device = device.lower()
        # check anyway
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = requested_device
        folder = False
        inf_path = Path(path)
        checkpoint_path = Path(checkpoint)
        config_path = Path(config)
        config = load_config(config_path)
        if confidence_threshold is None:
            confidence_threshold = config["postprocess"].get("confidence_threshold", 0.0)
            
        if output == ".":
            output_path = inf_path
        else:
            output_path = output
        if not inf_path.exists():
            print(f"Unable to locate folder {str(inf_path)}")
            sys.exit(1)
        if inf_path.is_dir():
            folder = True
        if lang_id is not None and lang_id <= -1:
            lang_id = None

        if folder:
            infer_folder(
                folder_path=str(inf_path),
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                output_dir=str(output_path),
                device=device,
                lang_id=lang_id,
                sample=sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                confidence_threshold=confidence_threshold
            )
        else:
            segments = infer_audio(
                audio_path=str(inf_path),
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                output_lab_path=str(output_path),
                device=device,
                lang_id=lang_id,
                sample=sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                confidence_threshold=confidence_threshold
            )
            print("Predicted segments:")
            for seg in segments:
                start, end, ph = seg
                print(f"({round(start, 2)}, {round(end, 2)}, {ph})")
    main()
