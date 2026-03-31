# Task 2 and Specific Task 2


## Task 2: Jets as Graphs (GNN Classifier)

### What I did

Converted jet images into graphs and trained a Graph Neural Network to classify them as quark or gluon.

Instead of treating the event as a 125x125 image, each non-zero pixel becomes a node in a graph. Nodes connect to their nearest neighbours. The GNN learns from the structure of those connections.

### Dataset

Same as Task 1. 5000 events, 3 channels (Tracks, ECAL, HCAL), labels 0 = gluon, 1 = quark.
Split: 4000 train / 500 val / 500 test.

### Code walkthrough

The code does three things in order.

**Step 1: Image to point cloud.**
For each event, only non-zero pixels are kept. Each pixel becomes a point with 5 features: row position, column position, and energy in each of the 3 channels. Most pixels are zero in jet events so this reduces the data significantly.

**Step 2: Point cloud to graph.**
Each point becomes a node. Edges connect each node to its 4 nearest neighbours based on position. This gives the GNN local geometric context, not just individual pixel values.

**Step 3: Train the GNN.**
The model is EdgeConv, which looks at each node and its neighbours and learns from how they relate to each other. Three stacked layers build up from local pixel relationships to global jet structure. A classifier head gives the final quark/gluon prediction.

```
Input nodes: (row, col, track, ecal, hcal) = 5 features
EdgeConv Layer 1:  5  -> 64
EdgeConv Layer 2:  64 -> 128
EdgeConv Layer 3:  128 -> 256
Global Pooling (mean + max) -> 512
Classifier -> 2 outputs (quark / gluon)
```

Training runs for up to 30 epochs. Loss is CrossEntropy. Optimizer is Adam.

### Problems I ran into

**Overfitting.** In the first run, train accuracy hit 94% while val accuracy stayed at 64%. The model was memorising the training data instead of learning general patterns.

**Fix: early stopping with higher regularisation.** The model now watches validation AUC after every epoch. If it does not improve for 5 epochs in a row, training stops and the best weights are loaded back. Dropout was also increased to 0.5. This brought training under control and the train/val loss curves stayed close together.

### Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 0.6280 |
| Test ROC-AUC | 0.6725 |

### Outputs

<img width="586" height="489" alt="image" src="https://github.com/user-attachments/assets/18bb43d6-ad30-450f-886d-bad7f43a0b3c" />


The ROC curve is above the random baseline line. AUC of 0.67 is reasonable for a small dataset of 5000 events. The literature reports 0.70 to 0.78 for GNNs on this dataset with much larger training sets.

### Architecture discussion

EdgeConv was chosen because it naturally handles point cloud data. It does not assume a fixed grid and works directly on the local neighbourhood of each node. This makes it a better fit for jet events than a standard CNN, since the energy deposits are sparse and irregular.

The main limitation is the cap of 100 nodes per graph. I kept the 100 highest energy pixels to save memory. This discards some low energy deposits that could carry useful information. With more compute, removing this cap would likely improve AUC.



## Specific Task 2: INR for Jet Event Representation

### What I did

Represented each jet event as a continuous function using an Implicit Neural Representation (INR).

Instead of storing the event as a grid of pixels, a small neural network is fit to it. The network takes a coordinate (eta, phi) and outputs the energy at that location across all 3 channels. The whole event is encoded in the network weights.

### Why this is interesting

A normal image is fixed resolution. You cannot query it between pixels. An INR has no fixed resolution. You can query the same trained network at 64x64, 125x125, or 250x250 without retraining.

### Code walkthrough

The code builds a coordinate grid once before any training. This grid maps every pixel position in the 125x125 image to a normalised (eta, phi) coordinate between -1 and 1. Building it once and reusing it for every event saves a lot of time.

The model is a Fourier Feature Network. A plain MLP struggles to learn sharp energy spikes because it smooths everything out. The fix is to encode the 2D coordinates through random sine and cosine functions before feeding them into the MLP. This lets the network learn fine detail without blurring.

```
Input: (eta, phi) coordinate
Fourier Encoding: 2D -> 256 features (sin + cos)
MLP: 256 -> 128 -> 128 -> 128 -> 3
Output: energy for Tracks, ECAL, HCAL at that coordinate
```

One INR is fit per event. 800 gradient steps with Adam. After fitting, the network can be queried at any resolution.

### Problems I ran into

**Problem 1: Shape mismatch error.**
The values tensor was coming out as shape (375, 3) instead of (15625, 3). This caused a RuntimeError during training. The cause was that numpy's `.T` on a reshaped array returns a non-contiguous array, and PyTorch misread its shape. Fixed by adding `.copy()` after the transpose, which forces a proper contiguous array.

**Problem 2: All black output images.**
The first version picked the first gluon and quark by index. Those events happened to have very little energy so the images came out completely black. Also using `vmax=1` for colour scaling washed out any faint signal. Fixed by sorting all events by total energy and picking the brightest ones, then scaling each channel's colour range to its actual max value.

### Results

| Channel | MSE | PSNR (dB) | SSIM |
|---------|-----|-----------|------|
| Tracks | 0.00000 | 56.87 | 0.9785 |
| ECAL | 0.00000 | 55.25 | 0.9778 |
| HCAL | 0.00001 | 54.16 | 0.9740 |
| **Overall** | **0.00000** | **55.42** | **0.9768** |

PSNR above 50 dB and SSIM above 0.97 across all channels means the INR is reconstructing the energy field very accurately.

### Outputs

**Side by side comparison: Gluon events**

<img width="1259" height="537" alt="image" src="https://github.com/user-attachments/assets/2c19186d-804e-4cfb-9535-f8000c9bb8bd" />


Left 3 columns are the originals. Right 3 are the INR reconstructions.

**Side by side comparison: Quark events**

<img width="1258" height="528" alt="image" src="https://github.com/user-attachments/assets/8b704b1d-3d7a-4f0b-9563-8af47f8d6ce6" />


**Resolution independence demo**

<img width="1256" height="324" alt="image" src="https://github.com/user-attachments/assets/aa983ae4-66de-42b4-95f0-c26d95e8b4a6" />


The same trained INR weights are queried at 64x64, 125x125, and 250x250. The energy structure stays consistent across all resolutions. This is something a regular image cannot do.



## How to run

Both files are standalone scripts. Run them in Google Colab with the dataset mounted at `/content/drive/MyDrive/ml4sci_dataset.hdf5`.

1. Open the script in Colab
2. Make sure GPU is enabled (Runtime > Change runtime type > T4 GPU)
3. Run all cells top to bottom

For Task 2, install PyTorch Geometric first:
```
!pip install torch-geometric --quiet
```
