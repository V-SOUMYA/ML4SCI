# SPECIFIC TASK 2 — INR for Jet Event Representation

import torch.nn as nn, torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


for c in range(3):
    mx = images[:, :, :, c].max()
    if mx > 0:
        images[:, :, :, c] /= mx

images = images.transpose(0, 3, 1, 2)   # (5000, 3, 125, 125)
print(f"Images: {images.shape}")


H, W = 125, 125
rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
COORDS = np.stack([
    rows.flatten() / 62.5 - 1.0,
    cols.flatten() / 62.5 - 1.0
], axis=1).astype(np.float32)                    
COORDS_T = torch.tensor(COORDS).to(device)


class FourierINR(nn.Module):
    """
    Maps (eta, phi) coordinates to energy values.
    Fourier encoding lets it learn sharp energy spikes.
    """
    def __init__(self, num_fourier=128, hidden=128, sigma=10.0):
        super().__init__()
        B = torch.randn(num_fourier, 2) * sigma
        self.register_buffer('B', B)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_fourier, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),           nn.ReLU(),
            nn.Linear(hidden, hidden),           nn.ReLU(),
            nn.Linear(hidden, 3),                nn.Sigmoid()
        )

    def forward(self, x):
        proj = 2 * np.pi * (x @ self.B.T)
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(feat)


def fit_inr(image_3ch, steps=800, lr=5e-4, batch=2048):
    """image_3ch: (3, 125, 125)"""
    
    values   = image_3ch.reshape(3, -1).T.copy()        # (15625, 3)
    values_t = torch.tensor(values, dtype=torch.float).to(device)

    model = FourierINR().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(steps):
        idx  = torch.randperm(len(COORDS_T), device=device)[:batch]
        pred = model(COORDS_T[idx])
        loss = F.mse_loss(pred, values_t[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model


def reconstruct(model, H=125, W=125):
    if H == 125 and W == 125:
        coords_t = COORDS_T
    else:
        r, c = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([
            r.flatten() / (H/2) - 1.0,
            c.flatten() / (W/2) - 1.0
        ], axis=1).astype(np.float32)
        coords_t = torch.tensor(coords).to(device)

    model.eval()
    with torch.no_grad():
        out = model(coords_t).cpu().numpy()   # (H*W, 3)
    return out.T.reshape(3, H, W)


total_energy = images.sum(axis=(1, 2, 3))
top_events   = np.argsort(total_energy)[::-1][:20]

gluon_ids   = [i for i in top_events if labels[i] == 0][:2]
quark_ids   = [i for i in top_events if labels[i] == 1][:2]
demo_ids    = gluon_ids + quark_ids
demo_labels = ["Gluon", "Gluon", "Quark", "Quark"]

demo_recons = []
for i, idx in enumerate(demo_ids):
    print(f"Fitting {demo_labels[i]} (event {idx})...")
    demo_recons.append(reconstruct(fit_inr(images[idx])))
    print("  done")
print("All demo events fitted.")



channel_names = ["Tracks", "ECAL", "HCAL"]

fig, axes = plt.subplots(4, 6, figsize=(16, 14))
fig.suptitle("Original (left 3) vs INR Reconstructed (right 3)", fontsize=13)

for row, (idx, label, recon) in enumerate(zip(demo_ids, demo_labels, demo_recons)):
    orig = images[idx]
    for c in range(3):
        

        vmax = max(orig[c].max(), recon[c].max(), 1e-6)

        axes[row, c].imshow(orig[c],    cmap="hot", vmin=0, vmax=vmax)
        axes[row, c].set_title(f"{label}\nOrig {channel_names[c]}", fontsize=8)
        axes[row, c].axis("off")

        axes[row, c+3].imshow(recon[c], cmap="hot", vmin=0, vmax=vmax)
        axes[row, c+3].set_title(f"{label}\nINR {channel_names[c]}", fontsize=8)
        axes[row, c+3].axis("off")

plt.tight_layout()
plt.savefig("inr_reconstruction.png", dpi=130)
plt.show()


m = fit_inr(images[demo_ids[0]])
orig_event = images[demo_ids[0]]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Same INR weights, different query resolution (ECAL channel)", fontsize=11)

vmax = orig_event[1].max() if orig_event[1].max() > 0 else 1e-6

for ax, img, title in zip(
    axes,
    [orig_event[1],
     reconstruct(m, 64,  64)[1],
     reconstruct(m, 125, 125)[1],
     reconstruct(m, 250, 250)[1]],
    ["Original 125x125", "INR at 64x64", "INR at 125x125", "INR at 250x250"]
):
    ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("inr_resolution_demo.png", dpi=130)
plt.show()



EVAL_N  = 30
mse_ch  = np.zeros(3)
psnr_ch = np.zeros(3)
ssim_ch = np.zeros(3)

for i in range(EVAL_N):
    orig  = images[i]
    recon = reconstruct(fit_inr(orig, steps=600))
    for c in range(3):
        mse  = np.mean((orig[c] - recon[c]) ** 2)
        mse_ch[c]  += mse
        psnr_ch[c] += 10 * np.log10(1.0 / (mse + 1e-10))
        ssim_ch[c] += ssim(orig[c], recon[c], data_range=1.0)
    if (i+1) % 10 == 0:
        print(f"Evaluated {i+1}/{EVAL_N}")

mse_ch /= EVAL_N
psnr_ch /= EVAL_N
ssim_ch /= EVAL_N

print(f"\n{'Channel':<10} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>8}")
print("-" * 44)
for c, name in enumerate(channel_names):
    print(f"{name:<10} {mse_ch[c]:>10.5f} {psnr_ch[c]:>12.2f} {ssim_ch[c]:>8.4f}")
print("-" * 44)
print(f"{'Overall':<10} {mse_ch.mean():>10.5f} {psnr_ch.mean():>12.2f} {ssim_ch.mean():>8.4f}")
