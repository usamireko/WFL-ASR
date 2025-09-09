import os
import json
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
import torchaudio
from model import BIOPhonemeTagger
from utils import decode_bio_tags, save_lab, load_phoneme_list, visualize_prediction, merge_adjacent_segments, load_langs, load_phoneme_merge_map, canonical_to_lang
from scipy.ndimage import median_filter
import argparse #finally lmao
import pytorch_optimizer as optim
import inspect


def collate_fn(batch):
    input_values = [item[0] for item in batch]
    label_ids = [item[1] for item in batch]
    wavs = [item[2] for item in batch]
    segments_gt = [item[3] for item in batch]
    wav_paths = [item[4] for item in batch]
    lang_ids = [item[5] for item in batch]

    label_lengths = torch.tensor([len(x) for x in label_ids])

    padded_input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    padded_label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)

    return padded_input_values, padded_label_ids, wavs, segments_gt, wav_paths, torch.tensor(lang_ids,
                                                                                              dtype=torch.long), label_lengths


class PhonemeDataset(Dataset):
    def __init__(self, dataset_path, label_list, max_seq_len=None, aug_cfg=None):
        with open(dataset_path, "r") as f:
            self.samples = json.load(f)
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_seq_len = max_seq_len
        self.aug_cfg = {
            "enable": False,
            "prob": 1.0,
            "noise_std": 0.0,
            "volume_range": [1.0, 1.0],
        }
        if aug_cfg:
            self.aug_cfg.update(aug_cfg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wav, sr = sf.read(sample["wav_path"])

        if sr != 16000:
            wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()

        max_amp = np.max(np.abs(wav))
        if max_amp > 0:
            wav = wav / max_amp
        else:
            wav = wav  # apparently it dies on silence

        if self.aug_cfg.get("enable", False) and random.random() < self.aug_cfg.get("prob", 1.0):
            scale = random.uniform(*self.aug_cfg.get("volume_range", [1.0, 1.0]))
            wav = wav * scale
            noise_std = self.aug_cfg.get("noise_std", 0.0)
            if noise_std > 0:
                wav = wav + np.random.normal(0.0, noise_std, wav.shape)
            wav = np.clip(wav, -1.0, 1.0)

        input_values = torch.tensor(wav, dtype=torch.float32)

        if self.max_seq_len:
            input_values = input_values[:self.max_seq_len]

        label_ids = [self.label2id.get(tag, self.label2id["O"]) for tag in sample["bio_tags"]]
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return input_values, label_ids, wav, sample["phoneme_segments"], sample["wav_path"], sample["lang_id"]

def clean_lab(ph_segment):
    if isinstance(ph_segment, (tuple, list)) and len(ph_segment) == 3:
        ph = ph_segment[2]
    else:
        ph = ph_segment
    while isinstance(ph, (tuple, list)) and len(ph) == 1:
        ph = ph[0]
    return str(ph).split("/")[-1]

def compute_framewise_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0

def compute_phoneme_error_rate(pred_segments, gt_segments):
    pred_seq = [ph for _, _, ph in pred_segments]
    gt_seq = [ph for _, _, ph in gt_segments]
    m, n = len(gt_seq), len(pred_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt_seq[i - 1] == pred_seq[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    edit_distance = dp[m][n]
    return edit_distance / max(m, 1)

def compute_timing_error(pred_segments, gt_segments):
    #print("[DEBUG] GT segments:", gt_segments[:5])
    #print("[DEBUG] Pred segments:", pred_segments[:5])
    matched_errors = []
    gt_durations = []

    for gt_start, gt_end, gt_ph in gt_segments:
        for pred_start, pred_end, pred_ph in pred_segments:
            if clean_lab(pred_ph) == clean_lab(gt_ph):
                start_error = abs(gt_start - pred_start)
                end_error = abs(gt_end - pred_end)
                matched_errors.append((start_error, end_error))
                gt_durations.append(gt_end - gt_start)
                break

    if not matched_errors or not gt_durations:
        return 0.0  # no match = no TER

    avg_timing_error = np.mean([e[0] + e[1] for e in matched_errors]) / 2
    avg_duration = np.mean(gt_durations)

    return avg_timing_error / avg_duration if avg_duration > 0 else 0.0

def compute_segmental_loss(segments_pred, segments_gt, loss_weights=(1.0, 1.0, 2.0)):
    w_start, w_end, w_iou = loss_weights
    total_loss = 0.0
    match_count = 0

    for seg in segments_gt:
        if not isinstance(seg, (list, tuple)) or len(seg) != 3:
            #print("Skipping malformed segment:", seg)
            continue
        gt_start, gt_end, gt_ph = seg

        best_score = float("inf")

        for pred_start, pred_end, pred_ph in segments_pred:
            if pred_ph != gt_ph:
                continue

            i_start = max(gt_start, pred_start)
            i_end = min(gt_end, pred_end)
            intersection = max(0.0, i_end - i_start)
            union = max(gt_end, pred_end) - min(gt_start, pred_start)
            iou = intersection / union if union > 0 else 0.0

            start_error = abs(gt_start - pred_start)
            end_error = abs(gt_end - pred_end)

            score = w_start * start_error + w_end * end_error + w_iou * (1.0 - iou)
            best_score = min(best_score, score)

        if best_score != float("inf"):
            total_loss += best_score
            match_count += 1

    if match_count == 0:
        return torch.tensor(0.0, requires_grad=True)

    return torch.tensor(total_loss / match_count, requires_grad=True)

def run_train_step(model, train_loader, optimizer, criterion, label_list, writer, step, config):
    frame_duration = config["data"].get("frame_duration", 0.02)
    model.train()

    for batch in train_loader:
        input_values, label_ids, wavs, segments_gt_batch, _, lang_ids, label_lengths = batch
        input_values = input_values.cuda()
        label_ids = label_ids.cuda()
        lang_ids = lang_ids.cuda()

        max_label_len = torch.max(label_lengths) if label_lengths.numel() > 0 else 0
        logits, offsets = model(input_values, lang_ids, max_label_len=max_label_len)

        loss = criterion(logits.view(-1, logits.size(-1)), label_ids.view(-1))

        total_segmental_loss = 0.0
        total_offset_loss = 0.0
        batch_size = input_values.size(0)

        id2label = {i: l for i, l in enumerate(label_list)}
        pred_ids_batch = torch.argmax(logits, dim=-1).cpu()

        for i in range(batch_size):
            pred_ids = pred_ids_batch[i, :label_lengths[i]].numpy()
            pred_tags = [id2label[p] for p in pred_ids]
            current_offsets = offsets[i, :label_lengths[i], :] if offsets is not None else None
            segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=current_offsets)

            segments_gt = segments_gt_batch[i]
            if isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
                segments_gt = segments_gt[0]

            segmental_loss = compute_segmental_loss(
                segments_pred, segments_gt,
                loss_weights=config["model"].get("segmental_loss_weights", (1.0, 1.0, 2.0))
            )
            total_segmental_loss += segmental_loss

            offset_loss = 0.0
            offset_count = 0
            for seg in segments_gt:
                if not isinstance(seg, (list, tuple)) or len(seg) != 3:
                    continue
                gt_start, gt_end, _ = seg
                start_frame = int(gt_start / frame_duration)
                end_frame = int(gt_end / frame_duration)
                start_offset_val = (gt_start / frame_duration) - start_frame
                end_offset_val = (gt_end / frame_duration) - end_frame

                if current_offsets is not None:
                    if start_frame < current_offsets.size(0):
                        pred_start = current_offsets[start_frame, 0]
                        offset_loss += torch.abs(pred_start - start_offset_val)
                        offset_count += 1
                    if end_frame < current_offsets.size(0):
                        pred_end = current_offsets[end_frame, 1]
                        offset_loss += torch.abs(pred_end - end_offset_val)
                        offset_count += 1
            
            if offset_count > 0:
                total_offset_loss += offset_loss / offset_count

        loss += config["model"].get("segmental_loss_weight", 1.0) * (total_segmental_loss / batch_size)
        loss += config["model"].get("subframe_loss_weight", 1.0) * (total_offset_loss / batch_size)
        writer.add_scalar("train/offset_loss", (total_offset_loss / batch_size).item() if isinstance(total_offset_loss, torch.Tensor) else total_offset_loss / batch_size, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        writer.add_scalar("train/loss", loss.item(), step)
        print(f"\r[Step {step}] Loss: {loss.item():.4f}", end="")

        if step % config["training"]["val_check_interval"] == 0:
            return step, True
        if step >= config["training"]["max_steps"]:
            break

    return step, False
    
def run_validation(model, val_loader, label_list, config, writer, step, best_loss, checkpoint_paths, criterion, id2lang, merge_map=None):
    val_loss = evaluate(model, val_loader, label_list, config, writer, step, criterion, id2lang, merge_map)

    model_path = os.path.join(config["output"]["save_dir"], f"model_step{step}.pt")
    torch.save(model.state_dict(), model_path)
    checkpoint_paths.append(model_path)

    max_ckpt = config["training"]["max_checkpoints"]
    if len(checkpoint_paths) > max_ckpt:
        to_remove = checkpoint_paths.pop(0)
        if os.path.exists(to_remove):
            os.remove(to_remove)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_path = os.path.join(config["output"]["save_dir"], "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"\nSaved best model with loss = {val_loss:.4f}")

    return best_loss

def train(config="config.yaml"):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    finetune_config = config.get("finetuning", {})
    enable_finetuning = finetune_config.get("enable", False)
    finetune_model_path = finetune_config.get("model_path", "")

    os.makedirs(config["output"]["save_dir"], exist_ok=True)

    phoneme_path = os.path.join(config["output"]["save_dir"], "phonemes.txt")
    dataset_path = os.path.join(config["output"]["save_dir"], "dataset.json")

    max_seq_conf = config["data"]["max_seq_len"]
    label_list = load_phoneme_list(phoneme_path)
    aug_cfg = config.get("augmentation", {})
    dataset = PhonemeDataset(dataset_path, label_list, max_seq_conf, aug_cfg)

    lang_map_path = os.path.join(config["output"]["save_dir"], "langs.txt")
    lang2id = load_langs(lang_map_path)
    id2lang = {i: l for l, i in lang2id.items()}

    merge_map_path = os.path.join(config["output"]["save_dir"], "phoneme_merge_map.json")
    merge_map = load_phoneme_merge_map(merge_map_path) if os.path.exists(merge_map_path) else None

    val_size = config["data"]["num_val_files"]
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], collate_fn=collate_fn, num_workers=config["training"]["num_workers"])

    model = BIOPhonemeTagger(config, label_list).cuda()

    if enable_finetuning and os.path.exists(finetune_model_path):
        print(f"[INFO] Loading finetune base model: {finetune_model_path}")
        state_dict = torch.load(finetune_model_path, map_location="cuda")

        old_langs = state_dict["lang_emb.weight"].shape[0]
        new_langs = config["model"]["num_languages"]
        if new_langs > old_langs:
            print(f"[INFO] Expanding lang_emb from {old_langs} -> {new_langs}")
            new_emb = nn.Embedding(new_langs, model.lang_emb.embedding_dim).cuda()
            new_emb.weight.data[:old_langs] = state_dict["lang_emb.weight"]
            new_emb.weight.data[old_langs:] = torch.randn_like(new_emb.weight[old_langs:]) * 0.01
            model.lang_emb = new_emb
            state_dict["lang_emb.weight"] = new_emb.weight

        base_phoneme_path = finetune_model_path.replace("best_model.pt", "phonemes.txt")
        if not os.path.exists(base_phoneme_path):
            raise RuntimeError(f"Missing phoneme list for base model: {base_phoneme_path}")
        old_label_list = load_phoneme_list(base_phoneme_path)
        old_label2id = {l: i for i, l in enumerate(old_label_list)}
        new_label2id = {l: i for i, l in enumerate(label_list)}

        print(f"[INFO] Attempting partial reuse of classifier weights: {len(old_label_list)} -> {len(label_list)}")

        new_out_dim = len(label_list)
        new_classifier = nn.Linear(model.classifier.in_features, new_out_dim).cuda()
        old_weight = state_dict["classifier.weight"]
        old_bias = state_dict["classifier.bias"]

        matched = 0
        for label in old_label_list:
            if label in new_label2id:
                old_idx = old_label2id[label]
                new_idx = new_label2id[label]
                new_classifier.weight.data[new_idx] = old_weight[old_idx]
                new_classifier.bias.data[new_idx] = old_bias[old_idx]
                matched += 1

        print(f"[INFO] Transferred weights for {matched} matching phoneme tags")

        model.classifier = new_classifier
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)

        model.load_state_dict(state_dict, strict=False)

    optimizer_name = config["training"].get("optimizer", "AdamW")
    optimizer_params = config["training"].get("optimizer_params", {})
    optimizer_params['lr'] = config["training"]["learning_rate"]
    
    if "weight_decay" in config["training"]:
        optimizer_params['weight_decay'] = config["training"]["weight_decay"]

    try:
        optimizer_class = getattr(optim, optimizer_name)
        print(f"[INFO] Using optimizer '{optimizer_name}' from pytorch-optimizer.")
    except AttributeError:
        print(f"[WARNING] Optimizer '{optimizer_name}' not found in pytorch-optimizer. Trying torch.optim.")
        try:
            import torch.optim as torch_optim
            optimizer_class = getattr(torch_optim, optimizer_name)
            print(f"[INFO] Using optimizer '{optimizer_name}' from torch.optim.")
        except AttributeError:
            print(f"[ERROR] Optimizer '{optimizer_name}' not found in torch.optim either. Please check your config.")
            return

    # Filter params for the optimizer
    sig = inspect.signature(optimizer_class.__init__)
    available_params = list(sig.parameters.keys())
    
    filtered_params = {k: v for k, v in optimizer_params.items() if k in available_params}

    optimizer = optimizer_class(
        model.parameters(),
        **filtered_params
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["val_check_interval"],
        gamma=config["training"].get("lr_decay_gamma", 0.5)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0), ignore_index=-100)
    writer = SummaryWriter(config["training"]["log_dir"])

    step = 0
    best_loss = float("inf")
    checkpoint_paths = []

    save_dir = config["output"]["save_dir"]
    checkpoint_files = sorted([
        f for f in os.listdir(save_dir) if f.startswith("model_step") and f.endswith(".pt")
    ], key=lambda x: int(x.replace("model_step", "").replace(".pt", "")))

    if checkpoint_files:
        last_ckpt = checkpoint_files[-1]
        ckpt_step = int(last_ckpt.replace("model_step", "").replace(".pt", ""))
        model.load_state_dict(torch.load(os.path.join(save_dir, last_ckpt)))
        step = ckpt_step
        print(f"Resumuing from checkpoint: {last_ckpt} (step {step})")

        checkpoint_paths = [os.path.join(save_dir, ckpt) for ckpt in checkpoint_files[-config["training"]["max_checkpoints"]:]]
        if os.path.exists(os.path.join(save_dir, "best_model.pt")):
            best_loss = float("inf")
    else:
        print("Training start")

    while step < config["training"]["max_steps"]:
        step, do_validate = run_train_step(model, train_loader, optimizer, criterion, label_list, writer, step, config)
        if do_validate:
            best_loss = run_validation(model, val_loader, label_list, config, writer, step, best_loss, checkpoint_paths, criterion, id2lang, merge_map)
            scheduler.step(step)
            new_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", new_lr, step)

    torch.save(model.state_dict(), os.path.join(config["output"]["save_dir"], "last_model.pt"))
    print("\nTraining complete at max_steps!")

def evaluate(model, val_loader, label_list, config, writer, step, criterion, id2lang, merge_map=None):
    model.eval()
    val_losses = []
    total_acc = 0.0
    total_per = 0.0
    total_ter = 0.0
    count = 0

    frame_duration = config["data"].get("frame_duration", 0.02)
    median_filter_size = config["postprocess"]["median_filter"]
    merge_segments = config["postprocess"]["merge_segments"]

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_values, label_ids, wavs, segments_gt_batch, _, lang_ids, label_lengths = batch
            input_values = input_values.cuda()
            label_ids = label_ids.cuda()
            lang_ids = lang_ids.cuda()

            max_label_len = torch.max(label_lengths) if label_lengths.numel() > 0 else 0
            logits, offsets = model(input_values, lang_ids, max_label_len=max_label_len)

            loss = criterion(logits.view(-1, logits.size(-1)), label_ids.view(-1))
            val_losses.append(loss.item())

            id2label = {i: l for i, l in enumerate(label_list)}
            pred_ids_batch = torch.argmax(logits, dim=-1).cpu()

            for j in range(input_values.size(0)):
                label_len = label_lengths[j]
                pred_ids = pred_ids_batch[j, :label_len].numpy()
                if median_filter_size > 1:
                    pred_ids = median_filter(pred_ids, size=median_filter_size)
                pred_tags = [id2label[p] for p in pred_ids]

                current_offsets = offsets[j, :label_len, :] if offsets is not None else None
                segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=current_offsets)

                if merge_segments != "none":
                    segments_pred = merge_adjacent_segments(segments_pred, mode=merge_segments)

                segments_gt = segments_gt_batch[j]
                if isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
                    segments_gt = segments_gt[0]

                acc = compute_framewise_accuracy(logits[j, :label_len].unsqueeze(0), label_ids[j, :label_len].unsqueeze(0))
                per = compute_phoneme_error_rate(segments_pred, segments_gt)
                ter = compute_timing_error(segments_pred, segments_gt)

                total_acc += acc
                total_per += per
                total_ter += ter
                count += 1

                if i == 0 and j == 0:  # Visualize first sample of first batch
                    lang_name = id2lang.get(lang_ids[j].item(), None)
                    vis_pred = segments_pred
                    vis_gt = segments_gt
                    if merge_map and lang_name:
                        vis_pred = [
                            (s, e, canonical_to_lang(ph, lang_name, merge_map))
                            for s, e, ph in segments_pred
                        ]
                        vis_gt = [
                            (s, e, canonical_to_lang(clean_lab(ph), lang_name, merge_map))
                            for s, e, ph in segments_gt
                        ]
                    
                    fig = visualize_prediction(
                        wavs[j],
                        config["data"]["sample_rate"],
                        vis_pred,
                        vis_gt,
                    )
                    writer.add_figure(f"val/prediction_{i}_{j}", fig, global_step=step)

    avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0
    avg_acc = total_acc / count if count > 0 else 0
    avg_per = total_per / count if count > 0 else 0
    avg_ter = total_ter / count if count > 0 else 0

    writer.add_scalar("val/loss", avg_loss, step)
    writer.add_scalar("val/accuracy", avg_acc, step)
    writer.add_scalar("val/per", avg_per, step)
    writer.add_scalar("val/ter", avg_ter, step)

    print(f"\n[Validation] Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | PER: {avg_per:.3f} | TER: {avg_ter:.3f}")
    return avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WFL model with a config file")
    parser.add_argument("config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()

    train(args.config)
