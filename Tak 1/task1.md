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

## Tasks completed

| Task | Description | Status |
|------|-------------|--------|
| Common Task 1 | Autoencoder for jet event reconstruction | Done |
| Common Task 2 | GNN for quark/gluon classification | Done |
| Specific Task 2 | INR-based event representation (main project) | Done |
| Specific Task 1 | Contrastive learning classifier | Done |

---

## Dataset

All tasks use the **CMS Quark/Gluon jet dataset**.

- Each event is a 125×125 image with 3 channels: Tracks, ECAL, HCAL
- Labels: 0 = gluon, 1 = quark
- Loaded from `ml4sci_dataset.hdf5` via Google Drive

---

## Task 1 — Autoencoder

**Goal:** compress a jet event image into a small vector, then reconstruct it back.

The encoder squishes `(3, 125, 125)` down to a 128-dimensional vector.
The decoder rebuilds the original image from that vector.

**Results on test set:**

| Channel | MSE | PSNR (dB) | SSIM |
|---------|-----|-----------|------|
| Tracks | 0.00000 | 69.06 | 0.9998 |
| ECAL | 0.00000 | 58.59 | 0.9986 |
| HCAL | 0.00001 | 51.07 | 0.9931 |
| **Overall** | **0.00000** | **55.08** | — |

PSNR above 40 dB is considered excellent. All three channels are well above that.

**Sample reconstruction:**

![reconstruction](task1_autoencoder/reconstruction_comparison.png)

---

## Task 2 — GNN Classifier

**Goal:** classify jets as quark or gluon using a graph neural network.

Instead of treating the event as an image, we convert it to a graph:
- Each non-zero pixel becomes a **node** with features `(η, φ, track energy, ECAL energy, HCAL energy)`
- Edges connect each node to its 4 nearest neighbours in (η, φ) space
- We use **EdgeConv** layers to classify the graph

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | see notebook |
| ROC-AUC | see notebook |

---

## Specific Task 2 — INR (main project)

**Goal:** represent each jet event as a continuous function using an Implicit Neural Representation.

Instead of storing the event as a grid of pixels, we fit a small neural network to it.
The network takes a coordinate `(η, φ)` and outputs the energy at that location.
Same weights, different resolution — you can query at 125×125 or 250×250 without retraining.

**Architecture:** Fourier Feature Network → 4-layer MLP → 3 output channels

**Why this matters:** conventional grid representations are fixed resolution and geometry-specific.
An INR is continuous, resolution-independent, and works across detector types.

**Sample reconstruction:**

![inr](task_specific2_inr/inr_reconstruction.png)

---

## Specific Task 1 — Contrastive Learning

**Goal:** learn useful jet representations without using labels, then classify quark vs gluon.

We use **SimCLR** — the model sees two augmented views of the same event and learns to recognise them as the same, while pushing apart views from different events.

After pre-training, we freeze the encoder and train a simple linear layer on top.

---

## How to run

Every task is a self-contained Jupyter notebook that runs on **Google Colab**.

1. Open the notebook in Colab
2. Set runtime to T4 GPU (Runtime → Change runtime type → T4 GPU)
3. Mount your Google Drive and point to your copy of `ml4sci_dataset.hdf5`
4. Run all cells top to bottom

No extra installs needed beyond what's in Cell 1 of each notebook.

---

## Setup

```python
# The only external package needed
!pip install energyflow    # only for Task 2 if using energyflow loader
```

Standard packages used: `torch`, `numpy`, `matplotlib`, `h5py`, `scikit-learn`, `scikit-image`

All are pre-installed on Colab.

---

## Contact

**Soumya Vajahhala**
soumyarao2810@gmail.com
Rutgers University, M.S. Computer Science

GSoC Organisation: ML4SCI
Project: Learning Parametrization with Implicit Neural Representations
Mentors: Sergei Gleyzer, Ali Hariri, Amal Saif
