import os
import json
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
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

        return input_values, label_ids, wav, sample["phoneme_segments"], sample["wav_path"]

def run_train_step(model, train_loader, optimizer, criterion, label_list, writer, step, config):
    model.train()
    for batch in train_loader:
        input_values, label_ids, wav, _, _ = batch
        input_values = input_values[0].cuda()
        label_ids = label_ids[0].cuda()

        logits = model(input_values)
        logits = logits.squeeze(0)
        min_len = min(logits.size(0), label_ids.size(0))
        loss = criterion(logits[:min_len], label_ids[:min_len])

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

def train():
    with open("config.yaml") as f:
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
    criterion = nn.CrossEntropyLoss()
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

    torch.save(model.state_dict(), os.path.join(config["output"]["save_dir"], "last_model.pt"))
    print("\nTraining complete at max_steps!")

def evaluate(model, val_loader, label_list, config, writer, step, criterion):
    model.eval()
    val_losses = []
    median_filter_size = config["postprocess"]["median_filter"]
    merge_segments = config["postprocess"]["merge_segments"]

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_values, label_ids, wav, segments_gt, _ = batch
            input_values = input_values[0].cuda()
            label_ids = label_ids[0].cuda()

            logits = model(input_values)
            logits = logits.squeeze(0)

            min_len = min(logits.size(0), label_ids.size(0))
            loss = criterion(logits[:min_len], label_ids[:min_len])
            val_losses.append(loss.item())

            id2label = {i: l for i, l in enumerate(label_list)}
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            if median_filter_size > 1:
                pred_ids = median_filter(pred_ids, size=median_filter_size)
            pred_tags = [id2label[i] for i in pred_ids]
            segments_pred = decode_bio_tags(pred_tags, frame_duration=frame_duration)
            if merge_segments != "none":
                segments_pred = merge_adjacent_segments(segments_pred, mode=merge_segments)

            if isinstance(segments_gt, list) and len(segments_gt) == 1 and isinstance(segments_gt[0], list):
                segments_gt = segments_gt[0]

            fig = visualize_prediction(wav[0], config["data"]["sample_rate"], segments_pred, segments_gt)
            writer.add_figure(f"val/prediction_{i}", fig, global_step=step)

    avg_loss = sum(val_losses) / len(val_losses)
    writer.add_scalar("val/loss", avg_loss, step)
    print(f"\n[Validation] Avg Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    train()
