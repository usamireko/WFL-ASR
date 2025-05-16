import os
import yaml
import json
import torch
import numpy as np
from utils import pad_or_trim, median_cluster_filter
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from itertools import cycle
from math import floor
from model import BoundaryPredictor

def collate_variable_length(batch):
    mels, spikes = zip(*batch)
    max_len = max(mel.shape[-1] for mel in mels)
    padded_mels = torch.stack([pad_or_trim(mel, max_len) for mel in mels])
    padded_spikes = torch.stack([pad_or_trim(spike, max_len) for spike in spikes])
    return padded_mels, padded_spikes

class BoundaryDataset(Dataset):
    def __init__(self, dataset_json):
        with open(dataset_json, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mel = torch.load(sample["mel"])
        spike = torch.load(sample["spike"])
        return mel, spike

def plot_prediction(mel, gt_spike, pred_spike, step, threshold=0.5):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")

    gt_indices = (gt_spike > threshold).nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(gt_indices, int):
        gt_indices = [gt_indices]

    filtered_gt = []
    last = -999
    for idx in sorted(gt_indices):
        if idx - last > 2:
            filtered_gt.append(idx)
            last = idx

    for t in filtered_gt:
        ax.axvline(x=t, color="green", linestyle="-", linewidth=1.5, alpha=0.5)

    pred_indices = (pred_spike > threshold).nonzero(as_tuple=False).squeeze().tolist()
    if isinstance(pred_indices, int):
        pred_indices = [pred_indices]
    filtered_pred = median_cluster_filter(sorted(pred_indices), min_distance=2)
    for t in filtered_pred:
        ax.axvline(x=t, color="red", linestyle="-", linewidth=1.5, alpha=0.5)

    ax.set_title(f"Boundary Prediction Step {step}")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bin")
    legend_labels = [
        plt.Line2D([], [], linestyle="none", marker='o', color="red", markersize=8, label="Pred"),
        plt.Line2D([], [], linestyle="none", marker='o', color="green", markersize=8, label="GT"),
    ]
    ax.legend(handles=legend_labels, loc="upper right", frameon=True, fancybox=True, framealpha=0.6)
    plt.tight_layout()
    return fig

def train(config_path):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BoundaryDataset(os.path.join(config["output"]["save_dir"], "dataset.json"))

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_variable_length
    )

    val_indices = np.random.RandomState(42).choice(len(dataset), size=config["data"]["num_val_files"], replace=False)
    val_samples = [dataset[i] for i in val_indices]

    model = BoundaryPredictor(
        mel_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(config["output"]["log_dir"])

    step = 0
    model.train()
    steps_per_epoch = len(dataset) // config["training"]["batch_size"]

    for mel, gt_spike in cycle(dataloader):
        current_epoch = floor(step / steps_per_epoch)

        mel = mel.to(device)
        gt_spike = gt_spike.to(device)
        mel = mel.squeeze(1) if mel.dim() == 4 else mel

        # Noise Augmentation <333 to hopefully help it generealize better
        if model.training:
            noise_std = config["training"].get("mel_noise_std", 0.05)
            mel = mel + torch.randn_like(mel) * noise_std

        out_main, out_mid1, out_mid2 = model(mel)

        target_len = gt_spike.shape[-1]
        if out_mid1.shape[-1] != target_len:
            out_mid1 = F.interpolate(out_mid1.unsqueeze(1), size=target_len, mode="linear", align_corners=True).squeeze(1)
        if out_mid2.shape[-1] != target_len:
            out_mid2 = F.interpolate(out_mid2.unsqueeze(1), size=target_len, mode="linear", align_corners=True).squeeze(1)

        loss_main = criterion(out_main, gt_spike)
        loss_mid1 = criterion(out_mid1, gt_spike)
        loss_mid2 = criterion(out_mid2, gt_spike)
        w1 = config["training"].get("inter_loss_weights", {}).get("mid1", 0.5)
        w2 = config["training"].get("inter_loss_weights", {}).get("mid2", 0.25)
        loss = loss_main + w1 * loss_mid1 + w2 * loss_mid2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        optimizer.step()

        if step % config["training"]["log_interval"] == 0:
            print(f"\r[Step {step}] [Epoch {current_epoch}] Loss: {loss.item():.4f}", end="", flush=True)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/loss_main", loss_main.item(), step)
            writer.add_scalar("train/loss_mid1", loss_mid1.item(), step)
            writer.add_scalar("train/loss_mid2", loss_mid2.item(), step)

        if step % config["training"]["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for i, (mel_val, gt_val) in enumerate(val_samples):
                    orig_len = mel_val.shape[-1]

                    if mel_val.dim() == 2:
                        mel_val = mel_val.unsqueeze(0)
                    mel_val_padded = pad_or_trim(mel_val.to(device), 1024)

                    if gt_val.dim() == 1:
                        gt_val = gt_val.unsqueeze(0)
                    gt_val = pad_or_trim(gt_val.to(device), 1024)

                    out_main, _, _ = model(mel_val_padded)
                    pred = out_main.sigmoid()

                    mel_val = mel_val[..., :orig_len]
                    gt_val = gt_val[..., :orig_len]
                    pred = pred[..., :orig_len]

                    fig = plot_prediction(
                        mel_val[0].cpu(),
                        gt_val[0].cpu(),
                        pred[0].cpu(),
                        step=step * 100 + i,
                        threshold=config["postprocess"].get("threshold", 0.5)
                    )
                    writer.add_figure(f"val/sample_{i}", fig, step)

                ckpt_path = os.path.join(config["output"]["save_dir"], f"step_{step}.pt")
                os.makedirs(config["output"]["save_dir"], exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

            model.train()

        step += 1
        if step > config["training"]["max_steps"]:
            break

    save_path = os.path.join(config["output"]["save_dir"], "model_final.pt")
    os.makedirs(config["output"]["save_dir"], exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")


if __name__ == "__main__":
    train("/content/drive/MyDrive/Boundary_predictor/config.yaml")
