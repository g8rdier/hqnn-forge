"""
examples/quick_start.py
========================
End-to-end demo: synthetic imbalanced dataset → PCA → HQNN → Focal Loss.

This script demonstrates a complete training loop using hqnn-forge.
Requires the `scikit-learn` and `matplotlib` optional dependencies:

    pip install "hqnn-forge[examples]"

Usage
-----
    python examples/quick_start.py

Expected output (approximate, values depend on random seed):
    Epoch  1/10 | Loss: 0.2843 | Acc: 0.9375 | PosFrac: 0.0625
    ...
    Training complete.  Final accuracy on hold-out set: ~90%
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.optim as optim

# ── hqnn-forge imports ────────────────────────────────────────────────────
from hqnn_forge.models          import HybridBinaryClassifier
from hqnn_forge.preprocessing   import PCANormalizer
from hqnn_forge.utils           import FocalLoss, compute_class_weights

# Optional: scikit-learn only used for synthetic data generation
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED              = 42
N_SAMPLES         = 500   # total dataset size (small for demo speed)
N_RAW_FEATURES    = 20    # raw feature dimensionality
N_PCA_COMPONENTS  = 8     # PCA output = n_qubits
N_QUBITS          = 8
N_LAYERS          = 2
BATCH_SIZE        = 16
N_EPOCHS          = 10
LR                = 0.02
MINORITY_FRACTION = 0.05  # ~5% fraud rate


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Generate synthetic imbalanced dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("  hqnn-forge  |  Quick-Start Demo")
print("=" * 60)
print()

if HAS_SKLEARN:
    print(f"[1/5] Generating synthetic data ({N_SAMPLES} samples, "
          f"imbalance ratio {1 - MINORITY_FRACTION:.0%}/{MINORITY_FRACTION:.0%}) …")
    X_raw, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_RAW_FEATURES,
        n_informative=10,
        n_redundant=5,
        weights=[1 - MINORITY_FRACTION, MINORITY_FRACTION],
        flip_y=0.01,
        random_state=SEED,
    )
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=SEED
    )
else:
    # Fallback: pure numpy synthetic data
    print("[1/5] scikit-learn not available — generating pure-numpy synthetic data …")
    n_train, n_test = int(N_SAMPLES * 0.8), int(N_SAMPLES * 0.2)
    X_train_np = rng.standard_normal((n_train, N_RAW_FEATURES))
    X_test_np  = rng.standard_normal((n_test, N_RAW_FEATURES))
    # Imbalanced labels
    y_train_np = (rng.random(n_train) < MINORITY_FRACTION).astype(int)
    y_test_np  = (rng.random(n_test)  < MINORITY_FRACTION).astype(int)


pos_rate = y_train_np.mean()
print(f"    Train positives: {pos_rate:.1%} | Train size: {len(y_train_np)}")
print()


# ---------------------------------------------------------------------------
# 2. PCA + normalisation (no sklearn)
# ---------------------------------------------------------------------------
print(f"[2/5] PCA reduction: {N_RAW_FEATURES}D → {N_PCA_COMPONENTS}D …")
pca = PCANormalizer(n_components=N_PCA_COMPONENTS, scale_to_pi=True)
X_train = pca.fit_transform(X_train_np)   # torch.Tensor, (n_train, 8)
X_test  = pca.transform(X_test_np)        # torch.Tensor, (n_test, 8)

y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test  = torch.tensor(y_test_np,  dtype=torch.float32)

print(f"    Explained variance (top {N_PCA_COMPONENTS}): "
      f"{pca.explained_variance_ratio_.sum() * 100:.1f}%")
print(f"    Feature range after tanh-π scaling: "
      f"[{X_train.min():.2f}, {X_train.max():.2f}]")
print()


# ---------------------------------------------------------------------------
# 3. Build model
# ---------------------------------------------------------------------------
print(f"[3/5] Building HybridBinaryClassifier "
      f"({N_QUBITS} qubits, {N_LAYERS} VQC layers) …")

model = HybridBinaryClassifier(
    n_input_features=N_PCA_COMPONENTS,
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    use_classical_encoder=False,  # input is already N_PCA_COMPONENTS == N_QUBITS
    device_name="default.qubit",  # use default.qubit for demo portability
    diff_method="parameter-shift",
    init_strategy="restricted",
)
print(f"    Total trainable parameters: {model.count_parameters()}")
print()


# ---------------------------------------------------------------------------
# 4. Loss, optimiser, class weights
# ---------------------------------------------------------------------------
loss_fn   = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Pre-compute class weights for informational use
cw = compute_class_weights(y_train)
print(f"[4/5] Imbalance info | neg weight: {cw[0]:.3f}, pos weight: {cw[1]:.3f}")
print()


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
print("[5/5] Training …")
print("-" * 60)

dataset_size = len(X_train)
indices      = torch.arange(dataset_size)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    perm       = torch.randperm(dataset_size)
    epoch_loss = 0.0
    n_batches  = 0

    t0 = time.perf_counter()
    for start in range(0, dataset_size, BATCH_SIZE):
        idx  = perm[start : start + BATCH_SIZE]
        xb   = X_train[idx]
        yb   = y_train[idx]

        optimizer.zero_grad()
        logits = model(xb).squeeze(-1)      # (batch,)
        loss   = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches  += 1

    elapsed = time.perf_counter() - t0

    # ── Epoch stats ───────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        train_probs = model.predict_proba(X_train)
        train_preds = (train_probs >= 0.5).long()
        train_acc   = (train_preds == y_train.long()).float().mean().item()
        avg_loss    = epoch_loss / n_batches

    print(
        f"Epoch {epoch:2d}/{N_EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {train_acc:.3f} | "
        f"Time: {elapsed:.1f}s"
    )

print("-" * 60)


# ---------------------------------------------------------------------------
# Hold-out evaluation
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    test_preds = model.predict(X_test, threshold=0.5)
    test_acc   = (test_preds == y_test.long()).float().mean().item()

    # Per-class accuracy
    neg_mask  = (y_test == 0)
    pos_mask  = (y_test == 1)
    neg_acc   = (test_preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else float("nan")
    pos_acc   = (test_preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else float("nan")

print()
print("Hold-out evaluation:")
print(f"  Overall accuracy : {test_acc:.3f}")
print(f"  Negative (maj.)  : {neg_acc:.3f}")
print(f"  Positive (min.)  : {pos_acc:.3f}  ← key metric for fraud detection")
print()
print("Demo complete.  See hqnn_forge/ for full library source.")
