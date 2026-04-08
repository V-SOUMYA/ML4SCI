# ML4SCI GENIE — GSoC 2026 Evaluation Tasks  
Soumya Vajahhala | Rutgers University  

---

##  Overview

This repository contains my solutions to the **ML4SCI GENIE GSoC 2026 evaluation tasks**.

The objective is to explore different representations of jet events from high-energy physics data and evaluate how well they preserve structure and support downstream tasks like classification.

I implemented and compared three approaches:

- **CNN Autoencoder** → grid-based representation  
- **Graph Neural Network (GNN)** → relational / point cloud representation  
- **Implicit Neural Representation (INR)** → continuous function-based representation  

---

##  Key Results

| Method        | PSNR (dB) | SSIM  | AUC   |
|--------------|----------|-------|-------|
| Autoencoder  | 55.08    | 0.999 | —     |
| GNN          | —        | —     | 0.6725|
| INR          | 55.42    | 0.976 | —     |

---

##  Task 1 — Autoencoder (Reconstruction)

A convolutional autoencoder was trained to learn a compact latent representation of jet events.

- Input: 125×125×3 jet images (Tracks, ECAL, HCAL)  
- Latent space: 128-dimensional  
- Objective: reconstruct the original image  

### Results

- **PSNR: 55.08 dB** → very high signal fidelity  
- **SSIM: ~0.999** → near-perfect structural preservation  
- Accurate reconstruction across all channels  

<img width="651" height="541" alt="image" src="https://github.com/user-attachments/assets/82ec55e6-8de6-4cb2-95eb-85687559235f" />

---

##  Task 2 — Graph Neural Network (Classification)

Jet images were converted into graphs by treating non-zero pixels as nodes and connecting them using nearest neighbors.

- Node features: position + energy values  
- Model: EdgeConv-based GNN  
- Task: classify quark vs gluon jets  

### Results

- **Test Accuracy: ~0.63**  
- **ROC-AUC: 0.6725**

The model captures relational structure and performs reasonably well given the dataset size.

<img width="586" height="489" alt="image" src="https://github.com/user-attachments/assets/18bb43d6-ad30-450f-886d-bad7f43a0b3c" />

---

##  Implicit Neural Representation (INR)

Each jet event was represented as a **continuous function** instead of a fixed grid.

- Input: (η, φ) coordinate  
- Output: energy (Tracks, ECAL, HCAL)  
- Model: Fourier Feature Network + MLP  

### Results

- **PSNR: ~55 dB**
- **SSIM: ~0.97**

INR achieves reconstruction quality comparable to the autoencoder despite being a continuous representation.

###  Key Advantage: Resolution Independence

The same trained INR can be queried at different resolutions (64×64, 125×125, 250×250) **without retraining**, while preserving structure.

<img width="1256" height="324" alt="image" src="https://github.com/user-attachments/assets/aa983ae4-66de-42b4-95f0-c26d95e8b4a6" />

---

##  Key Insights

- Autoencoders provide strong reconstruction with compact latent representations  
- GNNs capture relational and geometric structure for classification  
- INRs enable **continuous, resolution-independent representations** with competitive accuracy  

---

##  Repository Structure

```
ML4SCI/
│
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
├── task_specific1_contrastive/
│   ├── task_specific1_contrastive.ipynb
│   └── README.md
│
└── Ml4Sci.pdf
```


Each folder contains:
- Jupyter notebooks (with outputs)
- Visualizations and results
- Task-specific explanations

---

##  How to Run

- Open notebooks in **Google Colab**  
- Enable GPU (Runtime → T4 GPU recommended)

Install dependencies (for GNN):
pip install torch-geometric


Dataset path:
/content/drive/MyDrive/ml4sci_dataset.hdf5

---

## 🔗 Links

- Proposal: `Ml4Sci.pdf`  
- GitHub: https://github.com/V-SOUMYA  
