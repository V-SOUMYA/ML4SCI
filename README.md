# Learning Parametrization with Implicit Neural Representations

**Google Summer of Code 2026 • ML4SCI / GENIE**  
**175 hours • Intermediate/Advanced**

---

## Information

- **Name:** Soumya Vajahhala  
- **Email:** soumyarao2810@gmail.com  
- **Phone:** +1 332-254-9093  
- **GitHub:** https://github.com/V-SOUMYA  
- **University:** Rutgers University, New Brunswick, NJ  
- **Degree:** M.S. Computer Science (Expected 2026)  
- **Undergrad:** B.Tech CS with ML Specialization, JUET (CGPA 8.0/10)  
- **Time Zone:** EST (UTC−5)  
- **Mentors:** Sergei Gleyzer, Ali Hariri, Amal Saif  
- **Duration:** 175 hours (Standard)  

---

# Why I Want to Work on This

I am not a particle physicist. But this project is about something I find genuinely interesting: what if we stored data as a function instead of a grid?

That question started for me during my fruit-detection project. I built a CNN on 224x224 images and it worked well. But I kept thinking: the fruit does not exist as pixels. It is a shape in space. Pixels are just one way to sample it. What if we learned the shape directly?

That led me to NeRF, then SIREN, then Fourier Feature Networks. I built a small coordinate MLP just to test the idea. It worked.

This project is the same idea on data that actually matters. Jet events at the LHC are energy fields. Right now, they get stored as 125x125 grids. An INR stores the same field as a small neural network that you can query at any point. That is a cleaner, more flexible representation. I want to build it.

---

# Background

## The problem with fixed grids

Each LHC detector records the same jet event differently:

- ECAL and HCAL record energy on a fixed grid (like an image).
- Tracking detectors record particle paths as scattered 3D points.

The current approach converts everything to a 125x125 pixel image. It works, but there are three issues:

- **Fixed resolution.** You cannot query between pixels. Upsampling is always lossy.  
- **Sparse data.** Most pixels are zero. The grid wastes space on emptiness.  
- **One geometry.** A model trained on ECAL images does not transfer easily to tracking data.  

An INR avoids all three. The representation is a function, not a grid.

## What is an INR?

An INR is a neural network that maps coordinates to values:

```
fθ(η, φ) → E
```

The network takes a detector coordinate and outputs the energy there. The whole event lives in the weights θ.

The problem with a plain MLP is that it tends to smooth out sharp edges. A jet has a dense core with sharp energy spikes. The fix is a positional encoding:

```
γ(x) = [sin(2πBx), cos(2πBx)],   B ~ N(0, σ²)
```

This is called a Fourier Feature encoding.

## What I already know

- **PyTorch training loops.** Built MobileNetV2 classifier from scratch.  
- **Coordinate MLPs.** Implemented Fourier Feature Network.  
- **Production code.** Built tools at Accenture.  
- **Math.** Linear algebra and probability.  

---

# Technical Plan

## The central question

*Does an INR give a better representation than an autoencoder or a GNN?*

Metrics:

1. **Reconstruction quality**
2. **Classification accuracy**

---

## Step 1: Preparing the data

```python
import numpy as np

def image_to_pairs(image):
    rows, cols = np.nonzero(image)
    coords = np.stack([rows, cols], axis=1)
    values = image[rows, cols]

    n = len(rows)
    all_idx = np.argwhere(image == 0)
    chosen = all_idx[np.random.choice(len(all_idx), n)]
    zero_vals = np.zeros(n)

    coords = np.vstack([coords, chosen])
    values = np.hstack([values, zero_vals])

    coords = coords / 62.5 - 1.0

    return coords, values
```

**Note:** Including zero pixels ensures the model learns empty regions.

---

## Step 2: The INR model

```python
import torch
import torch.nn as nn
import numpy as np

class INRModel(nn.Module):
    def __init__(self):
        super().__init__()

        B = torch.randn(256, 2) * 10.0
        self.register_buffer('B', B)

        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def encode(self, x):
        proj = 2 * np.pi * (x @ self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        return self.net(self.encode(x)).squeeze(-1)
```

---

## Step 3a: Fitting one INR per event

```python
def fit_one_event(coords, values):
    model = INRModel()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    coords_t = torch.tensor(coords, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)

    for step in range(2000):
        idx = torch.randperm(len(coords_t))[:512]
        pred = model(coords_t[idx])
        loss = ((pred - values_t[idx]) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    weights = torch.cat([p.data.flatten() for p in model.parameters()])
    return weights
```

---

## Step 3b: Hyper-network

```python
class HyperNet(nn.Module):
    def __init__(self, n_weights):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, 256)
        )

        self.head = nn.Linear(256, n_weights)

    def forward(self, image):
        z = self.encoder(image)
        weights = self.head(z)
        return weights, z
```

---

## Step 4: Evaluation

### Reconstruction Metrics
- MSE  
- PSNR  
- SSIM  
- Visual comparison  

### Classification

```python
all_features, all_labels = [], []
for image, label in dataset:
    coords, values = image_to_pairs(image)
    feat = fit_one_event(coords, values)
    all_features.append(feat)
    all_labels.append(label)

clf = nn.Sequential(
    nn.Linear(feat.shape[0], 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)
```

### Ablations
- With vs without Fourier encoding  
- σ ∈ {1, 5, 10, 20}  
- Depth variations  

---

# Timeline

| Phase | Dates | Hours | Deliverable |
|------|------|------|------------|
| Community Bonding | May 1–26 | 10h | Setup |
| Phase 1 | May 27–Jun 16 | 30h | INR base |
| Phase 2 | Jun 17–Jul 7 | 35h | Hyper-network |
| Midterm | ~Jul 14 | 5h | Report |
| Phase 3 | Jul 8–Jul 28 | 30h | Classification |
| Phase 4 | Jul 29–Aug 18 | 30h | Ablations |
| Final | Aug 19–25 | 15h | Submission |
| Buffer | Throughout | 20h | Flex |

**Total: 175h**

---

# Deliverables

- INR model  
- Hyper-network  
- Reconstruction metrics  
- Classification results  
- Ablation study  
- Upsampling demo  
- Jupyter notebooks  
- README + documentation  
- Unit tests  
- Final report  

---

# About Me

## GENIE Codebase Observations

- Uses `energyflow` dataset  
- PyTorch-based pipelines  
- No INR baseline exists  

## Evaluation Tasks

### Autoencoder

| Channel | MSE | PSNR | SSIM |
|--------|----|-----|------|
| Tracks | 0.00000 | 69.06 | 0.9998 |
| ECAL | 0.00000 | 58.59 | 0.9986 |
| HCAL | 0.00001 | 51.07 | 0.9931 |

---

### GNN

| Metric | Score |
|------|------|
| Accuracy | 0.6800 |
| ROC-AUC | 0.7316 |

---

### INR

| Channel | MSE | PSNR | SSIM |
|--------|----|-----|------|
| Tracks | 0.00000 | 55.50 | 0.9716 |
| ECAL | 0.00000 | 55.06 | 0.9773 |
| HCAL | 0.00001 | 53.45 | 0.9713 |

---

## Projects

- Fruit Detection (99.6% accuracy)  
- Accenture GenAI tools  
- Coordinate MLP  

---

## Why I will finish this

The plan is conservative. Core work is done in Phases 1–2. Remaining phases are extensions.

---

## Long-term connection

Interested in scientific ML and representation learning for real-world physics problems.

---

# References

1. Tancik et al. (2020) Fourier Features  
2. Sitzmann et al. (2020) SIREN  
3. Mildenhall et al. (2020) NeRF  
4. Dupont et al. (2022) Functa  
5. ML4SCI GENIE Project  
