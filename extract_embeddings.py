"""Step 1b: per-frame RGB embeddings from a pretrained vision model (DINOv2).

Why: MediaPipe misses hands in ~25% of frames. A pretrained image model sees the whole frame
(hand shape, face, body), so it does not depend on landmark detection.

  python extract_embeddings.py --features features2 --out features2_emb
  # then train with the embeddings switched on (PowerShell):
  $env:ISL_EMB_DIR="features2_emb"
  python train_bilstm.py --features features2 --split seen --mode classify --seed 1 --out runs\\clsemb_s1

Uses the same frame selection (stride/max-frames) as extract_keypoints.py, so frames line up.
First run downloads the model (~90 MB for dinov2-small).
"""
import argparse, csv, os
from pathlib import Path
import numpy as np
import torch, cv2
from transformers import AutoModel

from extract_keypoints import iter_frames, VIDEO_EXT

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features2", help="folder with manifest.csv")
    ap.add_argument("--out", default="features2_emb")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--size", type=int, default=336, help="frame is resized to size x size (multiple of 14)")
    ap.add_argument("--stride", type=int, default=2, help="must match extract_keypoints")
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModel.from_pretrained(a.model).to(dev).to(dtype).eval()
    print("device:", dev, "| model:", a.model, "| size:", a.size)

    rows = list(csv.DictReader(open(os.path.join(a.features, "manifest.csv"), encoding="utf-8")))
    os.makedirs(a.out, exist_ok=True)
    mism, dim = 0, None

    @torch.no_grad()
    def embed(batch):
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(dev).to(dtype)
        h = model(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()

    for n, r in enumerate(rows, 1):
        out_path = Path(a.out) / Path(r["feature_path"]).name
        if out_path.exists():
            continue
        src = Path(r["source"])
        kind = "video" if src.suffix.lower() in VIDEO_EXT else "frames"
        embs, batch = [], []
        for fr in iter_frames(kind, src, a.stride, a.max_frames):
            fr = cv2.resize(fr, (a.size, a.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            batch.append((fr - MEAN) / STD)
            if len(batch) == a.batch:
                embs.append(embed(batch)); batch = []
        if batch:
            embs.append(embed(batch))
        if not embs:
            print("no frames for", src)
            continue
        e = np.concatenate(embs).astype(np.float16)
        dim = e.shape[1]
        if len(e) != len(np.load(r["feature_path"])):
            mism += 1  # data.py trims to the shorter one
        np.save(out_path, e)
        if n % 10 == 0 or n == len(rows):
            print(f"{n}/{len(rows)}", end="\r")
    print(f"\ndone -> {a.out}  (embedding dim {dim}; length mismatches vs keypoints: {mism})")
    if dim and dim != 768:
        print(f"NOTE: embedding dim is {dim}, set  $env:ISL_EMB_DIM={dim}  before training.")


if __name__ == "__main__":
    main()