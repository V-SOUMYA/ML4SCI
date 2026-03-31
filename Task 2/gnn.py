# TASK 2 — Jets as Graphs (GNN Classifier)

!pip install torch-geometric --quiet

from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


data   = h5py.File("/content/drive/MyDrive/ml4sci_dataset.hdf5", "r")
images = data["X_jets"][:5000].astype(np.float32)
labels = data["y"][:5000].astype(np.int64)

for c in range(3):
    mx = images[:, :, :, c].max()
    if mx > 0:
        images[:, :, :, c] /= mx

print(f"Images: {images.shape}")


def image_to_pointcloud(img):
    mask = img.sum(axis=2) > 0
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((1, 5), dtype=np.float32)
    row_norm = rows / 62.5 - 1.0
    col_norm = cols / 62.5 - 1.0
    energies = img[rows, cols]
    return np.column_stack([row_norm, col_norm, energies]).astype(np.float32)


def pointcloud_to_graph(points, label, k=4, max_nodes=100):
    if len(points) > max_nodes:
        top_idx = np.argsort(points[:, 2:].sum(axis=1))[::-1][:max_nodes]
        points  = points[top_idx]

    x   = torch.tensor(points, dtype=torch.float)
    pos = x[:, :2]
    N   = x.shape[0]
    k_actual = min(k, N - 1)

    diff  = pos.unsqueeze(0) - pos.unsqueeze(1)
    dists = (diff ** 2).sum(dim=2)
    dists.fill_diagonal_(float('inf'))
    _, knn_idx = dists.topk(k_actual, dim=1, largest=False)

    src = torch.arange(N).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    dst = knn_idx.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long))


print("Building graphs...")
graph_list = []
for i in range(len(images)):
    pts = image_to_pointcloud(images[i])
    graph_list.append(pointcloud_to_graph(pts, labels[i]))
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/5000")
print("Done.")


random.seed(42)
random.shuffle(graph_list)

train_loader = GeoLoader(graph_list[:4000], batch_size=64, shuffle=True)
val_loader   = GeoLoader(graph_list[4000:4500], batch_size=64)
test_loader  = GeoLoader(graph_list[4500:],     batch_size=64)
print("Train: 4000  Val: 500  Test: 500")


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class JetGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EdgeConv(MLP(2*5,   64))
        self.conv2 = EdgeConv(MLP(2*64, 128))
        self.conv3 = EdgeConv(MLP(2*128, 256))
        self.clf   = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, ei)
        x = self.conv2(x, ei)
        x = self.conv3(x, ei)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.clf(x)

model     = JetGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0
    probs_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
            correct  += (logits.argmax(1) == batch.y).sum().item()
            total    += batch.y.size(0)
            probs_all.append(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
            labels_all.append(batch.y.cpu().numpy())
    auc = roc_auc_score(np.concatenate(labels_all), np.concatenate(probs_all))
    return loss_sum / len(loader), correct / total, auc

best_auc   = 0
best_state = None
patience   = 5
no_improve = 0
train_losses, val_losses, val_aucs = [], [], []

for epoch in range(1, 31):
    tr_loss, tr_acc, _       = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_auc = run_epoch(val_loader,   train=False)

    train_losses.append(tr_loss)
    val_losses.append(vl_loss)
    val_aucs.append(vl_auc)

    if vl_auc > best_auc:
        best_auc   = vl_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/30  train_acc={tr_acc:.3f}  "
              f"val_acc={vl_acc:.3f}  val_auc={vl_auc:.4f}")

    if no_improve >= patience:
        print(f"Early stop at epoch {epoch}. Best val AUC: {best_auc:.4f}")
        break

# Load best model before testing
model.load_state_dict(best_state)

# Test 
_, test_acc, test_auc = run_epoch(test_loader, train=False)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")


model.eval()
probs_all, labels_all = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        probs_all.append(torch.softmax(model(batch), 1)[:, 1].cpu().numpy())
        labels_all.append(batch.y.cpu().numpy())

fpr, tpr, _ = roc_curve(np.concatenate(labels_all), np.concatenate(probs_all))

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"EdgeConv GNN (AUC = {test_auc:.4f})")
plt.plot([0,1],[0,1],'k--', lw=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Quark/Gluon GNN")
plt.legend()
plt.tight_layout()
plt.savefig("gnn_roc_curve.png", dpi=120)
plt.show()



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label="Train")
axes[0].plot(val_losses,   label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(val_aucs, color='steelblue')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
axes[1].set_title("Validation AUC")

plt.tight_layout()
plt.savefig("gnn_training_curves.png", dpi=120)
plt.show()
