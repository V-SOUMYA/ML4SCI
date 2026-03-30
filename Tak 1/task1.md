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

The code is one continuous script. Here is what it does in order.

First it loads the HDF5 file from Google Drive and reads the first 5000 events. Each channel is then scaled to [0, 1] by dividing by its max value. The shape changes from `(5000, 125, 125, 3)` to `(5000, 3, 125, 125)` because PyTorch expects channels first.

One gluon and one quark are plotted across all 3 channels so you can see what the raw data looks like before training.

The data is split into train, val, and test sets. DataLoaders feed the model images in batches of 32 during training.

The autoencoder has two parts:
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

Training runs for 30 epochs. Each epoch feeds batches into the model, computes MSE loss between the reconstruction and the original, and updates the weights. The same is done on validation data without updating anything. Loss is printed every 5 epochs.

After training, a batch from the test set is run through the model and plotted as a side by side comparison. Left 3 columns are the originals, right 3 are the reconstructions.

Three metrics are then computed on the full test set:
- **MSE**: average pixel error. Lower is better.
- **PSNR**: signal quality in decibels. Higher is better. Above 40 dB is excellent.
- **SSIM**: structural similarity from 0 to 1. Above 0.95 is very good.

Finally, the trained model weights are saved to Google Drive.

---

## Raw events

<img width="900" height="590" alt="image" src="https://github.com/user-attachments/assets/8b103158-1a3b-433f-b66a-7f81b0bb8e5b" />


One gluon (top row) and one quark (bottom row) across all 3 channels.
HCAL shows the most visible structure. The square blocks are how the hadronic calorimeter records energy.
Most of the image is black. Jet events are sparse by nature, so that is expected.

---

## Reconstruction: Gluon

<img width="651" height="541" alt="image" src="https://github.com/user-attachments/assets/82ec55e6-8de6-4cb2-95eb-85687559235f" />

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
