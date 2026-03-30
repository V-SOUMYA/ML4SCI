# ML4SCI GENIE — GSoC 2026

This repo contains my solutions to the ML4SCI GENIE evaluation tasks for Google Summer of Code 2026.
The project I am applying for is **Learning Parametrization with Implicit Neural Representations**.

---

## What's in here

```
├── task1_autoencoder/
│   ├── task1_autoencoder.ipynb
│   ├── raw_events.png
│   ├── training_curve.png
│   ├── reconstruction_comparison.png
│   └── README.md
│
├── task2_gnn/
│   ├── task2_gnn.ipynb
│   └── README.md
│
├── task_specific2_inr/
│   ├── task_specific2_inr.ipynb
│   └── README.md
│
└── task_specific1_contrastive/
    ├── task_specific1_contrastive.ipynb
    └── README.md
```

---

# Task 1 — Autoencoder for Quark/Gluon Jet Events
**ML4SCI GENIE | GSoC 2026 | Soumya Vajahhala**

---

## What I did

Trained a convolutional autoencoder on 5000 jet events.
The model compresses each 125x125 image into 128 numbers, then rebuilds it back.
If the reconstruction looks close to the original, the model has learned the structure of the jet.

---

## Dataset

- File: `ml4sci_dataset.hdf5`
- Keys used: `X_jets` (images), `y` (labels)
- Shape: `(5000, 125, 125, 3)` with 3 channels: Tracks, ECAL, HCAL
- Labels: 0 = gluon, 1 = quark
- Split: 4000 train / 500 val / 500 test

---

## Code walkthrough

**Cell 1: Load data**
Opens the HDF5 file from Google Drive and reads the first 5000 events.
5000 is enough to train on and keeps things fast on Colab.

**Cell 2: Normalize**
Each channel is scaled to [0, 1] by dividing by its max value.
Shape goes from `(5000, 125, 125, 3)` to `(5000, 3, 125, 125)` because PyTorch expects channels first.

**Cell 3: Visualize raw events**
Picks one gluon and one quark and plots all 3 channels side by side.
Just to see what the data looks like before any training.

**Cell 4: DataLoaders**
Splits data into train, val, and test.
Wraps them in DataLoaders so the model sees images in batches of 32 during training.

**Cell 5: Autoencoder model**
Two parts:
- Encoder: 3 conv layers that shrink the image down to a 128-number vector
- Decoder: rebuilds the image from those 128 numbers back to 125x125

```
Encoder:
  Conv(3 to 16) + ReLU + MaxPool   ->  (16, 62, 62)
  Conv(16 to 32) + ReLU + MaxPool  ->  (32, 31, 31)
  Conv(32 to 64) + ReLU + MaxPool  ->  (64, 15, 15)
  Flatten + Linear                 ->  128-dim vector

Decoder:
  Linear + Reshape                 ->  (64, 15, 15)
  ConvTranspose + ReLU             ->  (32, 30, 30)
  ConvTranspose + ReLU             ->  (16, 60, 60)
  Upsample + Conv + Sigmoid        ->  (3, 125, 125)
```

Total parameters: **3,765,955**

**Cell 6: Training**
Runs 30 epochs. Each epoch does three things:
1. Feeds a batch of images into the autoencoder
2. Computes how different the reconstruction is from the original using MSE loss
3. Updates model weights to reduce that error

Then checks the same on validation data without updating anything.
Prints loss every 5 epochs.

**Cell 7: Loss curve**
Plots train and val loss over 30 epochs.
Both lines should drop and flatten. That means the model converged.

**Cell 8: Side by side comparison**
Takes a batch from the test set, runs it through the model, and plots original vs reconstructed.
Left 3 columns are the originals. Right 3 columns are what the model rebuilt.

**Cell 9: Metrics**
Computes 3 numbers across the test set to measure how good the reconstruction is:
- **MSE**: average pixel error. Lower is better.
- **PSNR**: signal quality in decibels. Higher is better. Above 40 dB is excellent.
- **SSIM**: how similar the structure looks, from 0 to 1. Above 0.95 is very good.

**Cell 10: Save**
Saves the trained model weights to Google Drive so nothing is lost if Colab disconnects.

---

## Raw events

![Raw Events](raw_events.png)

One gluon (top row) and one quark (bottom row) across all 3 channels.
HCAL shows the most visible structure. The square blocks are how the hadronic calorimeter records energy.
Most of the image is black. Jet events are sparse by nature, so that is expected.

---

## Reconstruction: Gluon

![Gluon Reconstruction](reconstruction_gluon.png)

Left 3 columns are the originals. Right 3 columns are the reconstructed versions.

---

## Reconstruction: Quark

![Quark Reconstruction](reconstruction_quark.png)

Left 3 columns are the originals. Right 3 columns are the reconstructed versions.

---

## Results

| Channel | MSE | PSNR (dB) | SSIM |
|---------|-----|-----------|------|
| Tracks | 0.00000 | 69.06 | 0.9998 |
| ECAL | 0.00000 | 58.59 | 0.9986 |
| HCAL | 0.00001 | 51.07 | 0.9931 |
| **Overall** | **0.00000** | **55.08** | |

- MSE is near 0 across all channels. The pixel level error is tiny.
- PSNR above 40 dB is considered excellent. All three channels are well above that.
- SSIM above 0.99 means the structure and shape of energy deposits is preserved very well.
- HCAL scores slightly lower than the other two. It has the coarsest patterns and is a bit harder to reconstruct perfectly.

---
