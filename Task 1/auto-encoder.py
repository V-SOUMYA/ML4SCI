import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from skimage.metrics import structural_similarity as ssim

# Dataset 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

from google.colab import drive
drive.mount('/content/drive')

data_path = "/content/drive/MyDrive/ml4sci_dataset.hdf5"

data = h5py.File(data_path, "r")
print(list(data.keys()))

images = data["X_jets"][:5000]   # only first 5000 samples
labels = data["y"][:5000]

print(images.shape)
print(labels.shape)

images = images[:5000]   # small dataset for speed

# -------


images = images.astype(np.float32)
 
for c in range(3):
    ch_max = images[:, :, :, c].max()
    if ch_max > 0:
        images[:, :, :, c] /= ch_max
 
# PyTorch wants (N, C, H, W) 
images = images.transpose(0, 3, 1, 2)   # (5000, 3, 125, 125)
print(f"After transpose: {images.shape}")
print(f"Value range: [{images.min():.2f}, {images.max():.2f}]")


channel_names = ["Tracks", "ECAL", "HCAL"]
 
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
fig.suptitle("Raw Events — Gluon (top) vs Quark (bottom)", fontsize=13)
 
gluon_idx = np.where(labels == 0)[0][0]
quark_idx  = np.where(labels == 1)[0][0]
 
for c in range(3):
    axes[0, c].imshow(images[gluon_idx, c], cmap="hot")
    axes[0, c].set_title(f"Gluon — {channel_names[c]}")
    axes[0, c].axis("off")
 
    axes[1, c].imshow(images[quark_idx, c], cmap="hot")
    axes[1, c].set_title(f"Quark — {channel_names[c]}")
    axes[1, c].axis("off")
 
plt.tight_layout()
plt.savefig("raw_events.png", dpi=120)
plt.show()


X = torch.tensor(images)
y = torch.tensor(labels.astype(np.float32))
 
dataset = TensorDataset(X, y)
 
n_total = len(dataset)
n_train = int(0.8 * n_total)   # 4000
n_val   = int(0.1 * n_total)   #  500
n_test  = n_total - n_train - n_val  # 500
 
train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
 
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)
 
print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")


class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
 
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
 
            # 31x31 -> 15x15
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
 
            nn.Flatten(),
            nn.Linear(64 * 15 * 15, 128),
            nn.ReLU()
        )
 
    def forward(self, x):
        return self.net(x)
 
 
class Decoder(nn.Module):
    # Reconstructs vector of size 128 to (3, 125, 125)
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64 * 15 * 15)
 
        self.net = nn.Sequential(
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
 
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
 
            
            nn.Upsample(size=(125, 125), mode='bilinear', align_corners=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()   # output in [0, 1]
        )
 
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 15, 15)
        return self.net(x)
 
 
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
 
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
 
 
model = Autoencoder().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()   # for mixed precision

train_losses, val_losses = [], []
NUM_EPOCHS = 30

for epoch in range(1, NUM_EPOCHS + 1):

    # Training
    model.train()
    train_loss = 0
    for X_batch, _ in train_loader:
        X_batch = X_batch.to(device)

        with torch.cuda.amp.autocast():          
            recon, _ = model(X_batch)
            loss = F.mse_loss(recon, X_batch)

        optimizer.zero_grad(set_to_none=True)    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            with torch.cuda.amp.autocast():
                recon, _ = model(X_batch)
                val_loss += F.mse_loss(recon, X_batch).item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}")

print("Training done.")


model.eval()
 
# Grab one batch from the test set
X_test_batch, y_test_batch = next(iter(test_loader))
 
with torch.no_grad():
    recon_batch, _ = model(X_test_batch.to(device))
 
recon_np = recon_batch.cpu().numpy()
orig_np  = X_test_batch.numpy()
y_np     = y_test_batch.numpy()
 

gluon_ids = np.where(y_np == 0)[0][:2]
quark_ids  = np.where(y_np == 1)[0][:2]
show_ids   = list(gluon_ids) + list(quark_ids)
row_labels = ["Gluon", "Gluon", "Quark", "Quark"]
 
fig, axes = plt.subplots(4, 6, figsize=(16, 14))
fig.suptitle("Original (left 3 columns) vs Reconstructed (right 3 columns)",
             fontsize=13)
 
for row, (idx, label) in enumerate(zip(show_ids, row_labels)):
    for c in range(3):
        
        axes[row, c].imshow(orig_np[idx, c], cmap="hot", vmin=0, vmax=1)
        axes[row, c].set_title(f"{label}\nOrig {channel_names[c]}", fontsize=8)
        axes[row, c].axis("off")
 
        
        axes[row, c+3].imshow(recon_np[idx, c], cmap="hot", vmin=0, vmax=1)
        axes[row, c+3].set_title(f"{label}\nRecon {channel_names[c]}", fontsize=8)
        axes[row, c+3].axis("off")
 
plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=130)
plt.show()


all_orig  = []
all_recon = []
 
model.eval()
with torch.no_grad():
    for X_batch, _ in test_loader:
        recon, _ = model(X_batch.to(device))
        all_orig.append(X_batch.numpy())
        all_recon.append(recon.cpu().numpy())
 
all_orig  = np.concatenate(all_orig)   # (500, 3, 125, 125)
all_recon = np.concatenate(all_recon)
 
print(f"{'Channel':<10} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>8}")
print("-" * 44)
 
for c, name in enumerate(channel_names):
    orig_c  = all_orig[:, c]    # (500, 125, 125)
    recon_c = all_recon[:, c]
 
    mse_val  = np.mean((orig_c - recon_c) ** 2)
    psnr_val = 10 * np.log10(1.0 / (mse_val + 1e-10))
 
    
    ssim_scores = [
        ssim(orig_c[i], recon_c[i], data_range=1.0)
        for i in range(100)
    ]
    ssim_val = np.mean(ssim_scores)
 
    print(f"{name:<10} {mse_val:>10.5f} {psnr_val:>12.2f} {ssim_val:>8.4f}")
 
print("-" * 44)
mse_all  = np.mean((all_orig - all_recon) ** 2)
psnr_all = 10 * np.log10(1.0 / (mse_all + 1e-10))
print(f"{'Overall':<10} {mse_all:>10.5f} {psnr_all:>12.2f}")
