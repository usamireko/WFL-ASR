import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

snap_threshold_sec = 0.03

def detect_boundaries(y, sr, frame_length=512, hop_length=160, flux_threshold=0.1, delta_window=5):
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
    flux = np.pad(flux, (1,), mode="constant")
    flux = flux / np.max(flux)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta_mag = np.mean(np.abs(delta), axis=0)
    delta_mag = delta_mag / np.max(delta_mag)

    min_len = min(len(flux), len(delta_mag))
    flux = flux[:min_len]
    delta_mag = delta_mag[:min_len]

    combined = 0.5 * flux + 0.5 * delta_mag
    peaks, _ = scipy.signal.find_peaks(combined, height=flux_threshold, distance=delta_window)

    shift_frames = 1
    shifted_peaks = np.clip(peaks - shift_frames, 0, len(combined) - 1)
    times = librosa.frames_to_time(shifted_peaks, sr=sr, hop_length=hop_length)
    flux_times = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop_length)

    return times.tolist(), flux, delta_mag, flux_times

def correct_lab_boundaries(wav_path, predicted_boundaries, snap_threshold=snap_threshold_sec):
    lab_path = wav_path.replace(".wav", ".lab")
    snapped_boundaries = []
    original_boundaries = []

    if not os.path.exists(lab_path):
        return snapped_boundaries, original_boundaries

    with open(lab_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = parts
                start_sec = float(start) / 1e7
                end_sec = float(end) / 1e7
                original_boundaries.append((start_sec, end_sec, label))

                closest_start = min(predicted_boundaries, key=lambda t: abs(t - start_sec)) if len(predicted_boundaries) > 0 else start_sec
                closest_end = min(predicted_boundaries, key=lambda t: abs(t - end_sec)) if len(predicted_boundaries) > 0 else end_sec

                if abs(closest_start - start_sec) <= snap_threshold:
                    start_sec = closest_start
                if abs(closest_end - end_sec) <= snap_threshold:
                    end_sec = closest_end

                snapped_boundaries.append((start_sec, end_sec, label))
    
    return snapped_boundaries, original_boundaries

def visualize_audio_features(wav_path, y, sr, predicted_boundaries, flux, delta_mag, flux_times, 
                             snapped_boundaries=None, original_boundaries=None, save_path="features_plot.png"):
    fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axs[0].set_title("Original Label")
    axs[0].plot(np.linspace(0, len(y)/sr, len(y)), y, color="lightblue")

    axs[1].set_title("Spectral Flux + MFCC Delta")
    axs[1].plot(flux_times, flux, label="Flux", color="purple")
    axs[1].plot(flux_times, delta_mag, label="MFCC", color="orange")
    axs[1].legend()

    axs[2].set_title("Corrected Label Boundaries")
    axs[2].plot(np.linspace(0, len(y)/sr, len(y)), y, color="lightblue")

    for t in predicted_boundaries:
        axs[1].axvline(t, color="magenta", linestyle="--", linewidth=1, alpha=1)

    if original_boundaries:
        for start, end, label in original_boundaries:
            axs[0].axvline(end, color="red", linestyle="-", linewidth=1)
            axs[0].text((start + end) / 2, max(y) * 0.8, label, ha="center", fontsize=8, color="red")

    if snapped_boundaries:
        for start, end, label in snapped_boundaries:
            axs[2].axvline(end, color="green", linestyle="-", linewidth=1)
            axs[2].text((start + end) / 2, max(y) * 0.8, label, ha="center", fontsize=8, color="green")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    #print(f"Saved visual .png to {save_path}")

def write_lab(wav_path, snapped_boundaries, save_over=True, out_path=None):
    if out_path is None:
        lab_path = wav_path.replace(".wav", ".lab")
    else:
        lab_path = out_path

    with open(lab_path, "w") as f:
        for start, end, label in snapped_boundaries:
            start_100ns = int(start * 1e7)
            end_100ns = int(end * 1e7)
            f.write(f"{start_100ns} {end_100ns} {label}\n")
    #print(f"corrected .lab file written to: {lab_path}")

def process_file(wav_path, save_plot=False):
    #print(f"Processing: {wav_path}")
    y, sr = librosa.load(wav_path, sr=16000)
    predicted_boundaries, flux, delta_mag, flux_times = detect_boundaries(y, sr)
    snapped_boundaries, original_boundaries = correct_lab_boundaries(wav_path, predicted_boundaries)

    write_lab(wav_path, snapped_boundaries)

    if save_plot:
        plot_path = wav_path.replace(".wav", ".png")
        visualize_audio_features(
            wav_path, y, sr,
            predicted_boundaries,
            flux, delta_mag, flux_times,
            snapped_boundaries, original_boundaries,
            save_path=plot_path
        )

def process_entry(entry, save_plot):
    process_file(entry, save_plot=save_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct .lab timing boundaries from audio features.",
        usage="%(prog)s <input_path> [--save_plot]"
    )
    parser.add_argument("input_path", type=str, help="Path to .wav file or folder containing .wav files")
    parser.add_argument("--save_plot", action="store_true", help="saves PNG visualization")

    args = parser.parse_args()
    input_path = args.input_path
    save_plot = args.save_plot

    if os.path.isdir(input_path):
        wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".wav")]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_entry, fp, save_plot) for fp in wav_files]
            with tqdm(total=len(futures)) as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
        print("\nLabel correction complete. All files processed.")

    elif input_path.endswith(".wav"):
        process_file(input_path, save_plot=save_plot)
    else:
        print("Wig snatched!!! Gimme a .wav file or folder- not this mess.")
