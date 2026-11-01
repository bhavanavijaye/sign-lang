import csv, os, random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_T = 160
# Gap filling for missed hand detections is OFF by default (no clear gain in tests).
# Turn on with:  $env:ISL_FILL_GAP=6   (PowerShell)
FILL_MAX_GAP = int(os.environ.get("ISL_FILL_GAP", "0"))

# Optional RGB frame embeddings made by extract_embeddings.py (one .npy per sample, same file name as
# the keypoint file). Turn on with:  $env:ISL_EMB_DIR="features2_emb"   (PowerShell)
EMB_DIR = os.environ.get("ISL_EMB_DIR", "")
EMB_DIM = int(os.environ.get("ISL_EMB_DIM", "768"))
KP_FEAT_DIM = 576
# $env:ISL_EMB_ONLY="1" -> use ONLY the RGB embeddings (no keypoints), to test how much signal they carry
EMB_ONLY = bool(EMB_DIR) and os.environ.get("ISL_EMB_ONLY", "") == "1"
FEAT_DIM = EMB_DIM if EMB_ONLY else KP_FEAT_DIM + (EMB_DIM if EMB_DIR else 0)


def read_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["feature_path"], r["sentence"]))
    return rows


def make_splits(rows, mode="seen", seed=42, val_frac=0.15, test_frac=0.15, n_unseen=10):
    """
    mode='seen'   : every sentence appears in train/val/test (different videos). Closed-set accuracy.
    mode='unseen' : `n_unseen` sentences are held out completely for test (real translation test),
                    remaining sentences are split train/val per-video.
    NOTE: without signer ids this is NOT signer-independent. If your filenames contain a signer id,
    split on that instead for a stricter test.
    """
    rng = random.Random(seed)
    by_sent = defaultdict(list)
    for r in rows:
        by_sent[r[1]].append(r)
    sents = sorted(by_sent)
    rng.shuffle(sents)
    train, val, test = [], [], []
    held = set(sents[:n_unseen]) if mode == "unseen" else set()
    for s in sents:
        items = by_sent[s][:]
        rng.shuffle(items)
        if s in held:
            test += items
            continue
        n = len(items)
        nv = max(1, round(n * val_frac)) if n >= 3 else 0
        nt = max(1, round(n * test_frac)) if (n >= 3 and mode == "seen") else 0
        val += items[:nv]
        test += items[nv:nv + nt]
        train += items[nv + nt:]
    return train, val, test


def resample(x, new_len):
    """Linear temporal resampling of any (T, ...) array."""
    idx = np.linspace(0, len(x) - 1, new_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (idx - lo).astype(np.float32).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[lo] * (1 - w) + x[hi] * w


def augment(x, e=None):
    """x: (T,75,3) normalized keypoints; e: optional (T,D) embeddings aligned with x.
    Temporal changes are applied to both; geometric changes only to keypoints."""
    T = len(x)
    new_T = int(np.clip(T * random.uniform(0.8, 1.25), 8, MAX_T))
    x = resample(x, new_T)
    if e is not None:
        e = resample(e, new_T)
    present = (x != 0).any(-1, keepdims=True)
    ang = np.deg2rad(random.uniform(-8, 8))
    c, s_ = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s_], [s_, c]], dtype=np.float32)
    x = x.copy()
    x[..., :2] = x[..., :2] @ R.T
    x *= random.uniform(0.9, 1.1)
    x[..., :2] += np.random.uniform(-0.05, 0.05, size=(1, 1, 2)).astype(np.float32)
    x += np.random.normal(0, 0.005, x.shape).astype(np.float32)
    x = x * present
    keep = np.random.rand(len(x)) > 0.05
    keep[0] = True
    x = x[keep]
    if e is not None:
        e = e[keep]
    return x, e


def hand_local(x):
    """Hand shape features: each hand centered on its wrist and scaled by hand size
    (wrist -> middle-finger base), so finger shapes are visible regardless of where the hand is.
    x: (T,75,3) -> (T,126). Missing hands stay 0."""
    outs = []
    for s in (33, 54):  # left hand, right hand
        h = x[:, s:s + 21, :]
        present = (h != 0).any(-1, keepdims=True)
        local = h - h[:, :1, :]
        size = np.linalg.norm(local[:, 9, :2], axis=-1)[:, None, None]
        size = np.where(size > 1e-4, size, 1.0)
        local = np.clip(local / size, -5, 5) * present
        outs.append(local.reshape(len(x), -1))
    return np.concatenate(outs, -1)


def fill_gaps(x, max_gap=6):
    """Linearly interpolate hands over short runs of missed detections. x: (T,75,3)."""
    x = x.copy()
    for s in (33, 54):
        present = (x[:, s:s + 21] != 0).any((1, 2))
        idx = np.where(present)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for k in range(1, gap + 1):
                    w = k / (gap + 1)
                    x[a + k, s:s + 21] = x[a, s:s + 21] * (1 - w) + x[b, s:s + 21] * w
    return x


def to_features(x):
    """(T,75,3) -> (T,576): positions (225) + velocities (225) + hand-local shape (126)."""
    flat = x.reshape(len(x), -1)
    vel = np.zeros_like(flat)
    vel[1:] = flat[1:] - flat[:-1]
    return np.concatenate([flat, vel, hand_local(x)], -1).astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, sent = self.rows[i]
        x = np.load(path)
        e = None
        if EMB_DIR:
            epath = os.path.join(EMB_DIR, os.path.basename(path))
            e = np.load(epath).astype(np.float32)
            n = min(len(x), len(e))  # keep keypoints and embeddings aligned
            x, e = x[:n], e[:n]
        if FILL_MAX_GAP:
            x = fill_gaps(x, FILL_MAX_GAP)
        if self.train:
            x, e = augment(x, e)
        if len(x) > MAX_T:
            x = resample(x, MAX_T)
            e = resample(e, MAX_T) if e is not None else None
        if EMB_ONLY:
            f = e.astype(np.float32)
        else:
            f = to_features(x)
            if e is not None:
                f = np.concatenate([f, e], -1).astype(np.float32)
        return torch.from_numpy(f), sent


def collate(batch):
    feats, sents = zip(*batch)
    T = max(len(f) for f in feats)
    D = feats[0].shape[1]
    kp = torch.zeros(len(feats), T, D)
    mask = torch.zeros(len(feats), T, dtype=torch.long)
    for i, f in enumerate(feats):
        kp[i, :len(f)] = f
        mask[i, :len(f)] = 1
    return kp, mask, list(sents)