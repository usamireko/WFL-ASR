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
from utils import decode_bio_tags, save_lab, load_phoneme_list, visualize_prediction, merge_adjacent_segments
from scipy.ndimage import median_filter

frame_duration = 0.02 # ~20ms per frame

class PhonemeDataset(Dataset):
    def __init__(self, dataset_path, label_list, max_seq_len=None):
        with open(dataset_path, "r") as f:
            self.samples = json.load(f)
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_seq_len = max_seq_len

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
            wav = wav # apparently it dies on silence

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
    return str(ph).split("/")[-1].lower()

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

    for gt_start, gt_end, gt_ph in segments_gt:
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
    model.train()

    for batch in train_loader:
        input_values, label_ids, wav, segments_gt, _, lang_id = batch
        input_values = input_values[0].cuda()
        label_ids = label_ids[0].cuda()
        lang_id = torch.tensor([lang_id], dtype=torch.long).cuda()

        logits, offsets = model(input_values, lang_id)
        logits = logits.squeeze(0)
        offsets = offsets.squeeze(0)  # [T, 2]

        min_len = min(logits.size(0), label_ids.size(0))
        loss = criterion(logits[:min_len], label_ids[:min_len])
        loss = loss.mean()

        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        id2label = {i: l for i, l in enumerate(label_list)}
        pred_tags = [id2label[i] for i in pred_ids]
        segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration)

        if isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
            segments_gt = segments_gt[0]

        segmental_loss = compute_segmental_loss(
            segments_pred, segments_gt,
            loss_weights=config["model"].get("segmental_loss_weights", (1.0, 1.0, 2.0))
        )
        loss += config["model"].get("segmental_loss_weight", 1.0) * segmental_loss

        offset_loss = 0.0
        offset_count = 0

        for gt_start, gt_end, gt_ph in segments_gt:
            start_frame = int(gt_start / frame_duration)
            end_frame = int(gt_end / frame_duration)
            start_offset_val = (gt_start / frame_duration) - start_frame
            end_offset_val = (gt_end / frame_duration) - end_frame

            start_offset = torch.tensor([start_offset_val], device=offsets.device)
            end_offset = torch.tensor([end_offset_val], device=offsets.device)

            if start_frame < offsets.size(0):
                pred_start = offsets[start_frame, 0]
                offset_loss += torch.abs(pred_start - start_offset)
                offset_count += 1

            if end_frame < offsets.size(0):
                pred_end = offsets[end_frame, 1]
                offset_loss += torch.abs(pred_end - end_offset)
                offset_count += 1

        if offset_count > 0:
            offset_loss = offset_loss / offset_count
        else:
            offset_loss = torch.tensor(0.0, device=logits.device)
        offset_loss = offset_loss.mean()

        loss += config["model"].get("subframe_loss_weight", 1.0) * offset_loss
        writer.add_scalar("train/offset_loss", offset_loss.item(), step)

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
    
def run_validation(model, val_loader, label_list, config, writer, step, best_loss, checkpoint_paths, criterion):
    val_loss = evaluate(model, val_loader, label_list, config, writer, step, criterion)

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

    os.makedirs(config["output"]["save_dir"], exist_ok=True)

    phoneme_path = os.path.join(config["output"]["save_dir"], "phonemes.txt")
    dataset_path = os.path.join(config["output"]["save_dir"], "dataset.json")

    max_seq_conf = config["data"]["max_seq_len"]
    label_list = load_phoneme_list(phoneme_path)
    dataset = PhonemeDataset(dataset_path, label_list, max_seq_conf)

    val_size = config["data"]["num_val_files"]
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = BIOPhonemeTagger(config, label_list).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = StepLR(
        optimizer,
        step_size=config["training"]["val_check_interval"],
        gamma=config["training"].get("lr_decay_gamma", 0.5)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))
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
            best_loss = run_validation(model, val_loader, label_list, config, writer, step, best_loss, checkpoint_paths, criterion)
            scheduler.step(step)
            new_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", new_lr, step)

    torch.save(model.state_dict(), os.path.join(config["output"]["save_dir"], "last_model.pt"))
    print("\nTraining complete at max_steps!")

def evaluate(model, val_loader, label_list, config, writer, step, criterion):
    model.eval()
    val_losses = []
    total_acc = 0.0
    total_per = 0.0
    total_ter = 0.0
    count = 0

    median_filter_size = config["postprocess"]["median_filter"]
    merge_segments = config["postprocess"]["merge_segments"]

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_values, label_ids, wav, segments_gt, _, lang_id = batch
            input_values = input_values[0].cuda()
            label_ids = label_ids[0].cuda()
            lang_id = torch.tensor([lang_id], dtype=torch.long).cuda()

            logits, offsets = model(input_values, lang_id)
            logits = logits.squeeze(0)
            offsets = offsets.squeeze(0)

            min_len = min(logits.size(0), label_ids.size(0))
            loss = criterion(logits[:min_len], label_ids[:min_len])
            val_losses.append(loss.item())

            if isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
                segments_gt = segments_gt[0]

            id2label = {i: l for i, l in enumerate(label_list)}
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            if median_filter_size > 1:
                from scipy.ndimage import median_filter
                pred_ids = median_filter(pred_ids, size=median_filter_size)
            pred_tags = [id2label[i] for i in pred_ids]

            segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration, offsets=offsets)

            if merge_segments != "none":
                segments_pred = merge_adjacent_segments(segments_pred, mode=merge_segments)

            acc = compute_framewise_accuracy(logits[:min_len], label_ids[:min_len])
            per = compute_phoneme_error_rate(segments_pred, segments_gt)
            ter = compute_timing_error(segments_pred, segments_gt)

            total_acc += acc
            total_per += per
            total_ter += ter
            count += 1

            fig = visualize_prediction(wav[0], config["data"]["sample_rate"], segments_pred, segments_gt)
            writer.add_figure(f"val/prediction_{i}", fig, global_step=step)

    avg_loss = sum(val_losses) / len(val_losses)
    avg_acc = total_acc / count
    avg_per = total_per / count
    avg_ter = total_ter / count

    writer.add_scalar("val/loss", avg_loss, step)
    writer.add_scalar("val/accuracy", avg_acc, step)
    writer.add_scalar("val/per", avg_per, step)
    writer.add_scalar("val/ter", avg_ter, step)

    print(f"\n[Validation] Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}% | PER: {avg_per:.3f} | TER: {avg_ter:.3f}")
    return avg_loss

if __name__ == "__main__":
    train("/content/drive/MyDrive/WFL_10/config.yaml")
