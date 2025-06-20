# WFL-ASR: Whisper/WavLM for Phoneme Labeling

**WFL-ASR** is a configurable deep learning model designed for automatic phoneme segmentation using frame-level BIO tagging. It supports both Whisper and WavLM as audio encoders, and is structured for flexible and efficient training on phoneme-aligned datasets.

---

## How It Works

This model performs **frame-level phoneme labeling** using the BIO tag format (`B-`, `I-`, `O`).

### 1. Label Preprocessing
- `.lab` files define phoneme segments using HTK format.
- Each segment is converted into BIO tags aligned to time frames based on `frame_duration` (hardcoded to 20ms for Whisper compatibility).
- Tags are stored along with the audio path in a training JSON.

### 2. Feature Extraction
- **Whisper** or **WavLM** encoders process the audio waveform into frame-wise feature vectors.
  - Whisper uses fixed 20ms frame stride.
  - WavLM offers flexible windowing via HuBERT-style encoding.

### 3. Neural Architecture
The encoded features go through a stack of optional, configurable layers:

- `BiLSTM` - sequential modeling (optional)
- `Conformer Blocks` - long + short-term feature modeling
- `Dilated Conv Stack` - local context enhancement (optional)

### 4. Classification
- A linear layer maps each time step to a BIO tag.

### 5. Inference and Postprocessing
- Predict BIO tags from audio.
- Optional smoothing (median filtering) and merging for better boundary clarity.
- Convert tags back to `.lab` segments.

---

## Features

- Whisper/WavLM encoder support
- Frame-level BIO tag training
- Configurable architecture (BiLSTM, Conformer, Conv)
- HTK-compatible `.lab` output format
- Optional waveform augmentation via the `augmentation` config section

---

## Augmentation Options

The `config.yaml` file now includes an optional `augmentation` section used during training. When enabled it randomly applies volume scaling and Gaussian noise:

```yaml
augmentation:
  enable: true
  noise_std: 0.005      # standard deviation of Gaussian noise
  prob: 0.5             # probability to augment a sample
  volume_range: [0.9, 1.1]  # random scaling of audio volume
```

Disable augmentation by setting `enable: false`.
