"""Slide-style training: keypoints -> Bi-LSTM -> Transformer decoder -> English.

  python train_bilstm.py --features features --split seen --mode seq2seq --out runs\\bilstm_seq2seq
  python train_bilstm.py --features features --split unseen --n-unseen 10 --mode seq2seq --out runs\\bilstm_unseen
  python train_bilstm.py --features features --split seen --mode classify --out runs\\bilstm_cls   # baseline

Same splits, features and metrics as train.py, so the numbers go in one comparison table.
"""
import argparse, csv, json, math, os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from data import read_manifest, make_splits, SignDataset, collate
from model_bilstm import BiLSTMTransformer, BiLSTMClassifier, PAD, BOS, EOS, UNK
from train import metrics, norm  # same metrics as the T5 run


def build_vocab(train_sents):
    words = sorted({w for s in train_sents for w in norm(s).split()})
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + words
    return itos, {w: i for i, w in enumerate(itos)}


def to_ids(sents, stoi):
    seqs = [[BOS] + [stoi.get(w, UNK) for w in norm(s).split()] + [EOS] for s in sents]
    L = max(len(s) for s in seqs)
    t = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, :len(s)] = torch.tensor(s)
    return t


def ids_to_text(row, itos):
    out = []
    for t in row.tolist():
        if t == EOS:
            break
        if t in (PAD, BOS):
            continue
        out.append(itos[t])
    return " ".join(out)


@torch.no_grad()
def predict(model, loader, dev, mode, itos=None, train_sents=None):
    model.eval()
    refs, hyps = [], []
    for kp, mask, sents in loader:
        kp, mask = kp.to(dev), mask.to(dev)
        if mode == "seq2seq":
            ids = model.greedy(kp, mask)
            hyps += [ids_to_text(r, itos) for r in ids.cpu()]
        else:
            idx = model(kp, mask).argmax(-1).cpu().tolist()
            hyps += [train_sents[i] for i in idx]
        refs += sents
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--split", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--n-unseen", type=int, default=10)
    ap.add_argument("--mode", choices=["seq2seq", "classify"], default="seq2seq")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/bilstm")
    a = ap.parse_args()

    if a.mode == "classify" and a.split == "unseen":
        raise SystemExit("classify mode can only predict known sentences. Use --split seen for it.")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| mode:", a.mode)

    rows = read_manifest(os.path.join(a.features, "manifest.csv"))
    train, val, test = make_splits(rows, a.split, a.seed, n_unseen=a.n_unseen)
    train_sents = sorted({s for _, s in train})
    print(f"samples: train {len(train)} | val {len(val)} | test {len(test)} | train sentences {len(train_sents)}")
    if not val:
        raise SystemExit("Validation split is empty.")

    itos, stoi = build_vocab(train_sents)
    s2c = {s: i for i, s in enumerate(train_sents)}
    if a.mode == "seq2seq":
        model = BiLSTMTransformer(len(itos), hid=a.hid, d_model=a.hid).to(dev)
        print("vocab size:", len(itos))
    else:
        model = BiLSTMClassifier(len(train_sents), hid=a.hid).to(dev)
    print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))

    def tl(rows_, train_):
        return DataLoader(SignDataset(rows_, train_), batch_size=a.bs, shuffle=train_,
                          collate_fn=collate, num_workers=0)
    tr, va, te = tl(train, True), tl(val, False), tl(test, False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(tr)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    ce = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1) if a.mode == "seq2seq" \
        else nn.CrossEntropyLoss(label_smoothing=0.1)

    best, history = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for kp, mask, sents in tr:
            kp, mask = kp.to(dev), mask.to(dev)
            if a.mode == "seq2seq":
                tgt = to_ids(sents, stoi).to(dev)
                logits = model(kp, mask, tgt[:, :-1])
                loss = ce(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            else:
                y = torch.tensor([s2c[s] for s in sents], device=dev)
                loss = ce(model(kp, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
        msg = f"ep {ep:03d} loss {run / len(tr):.4f}"
        if ep % a.eval_every == 0 or ep == a.epochs:
            refs, hyps = predict(model, va, dev, a.mode, itos, train_sents)
            m = metrics(refs, hyps, train_sents)
            history.append({"epoch": ep, **m})
            msg += f" | val exact {m['exact_acc']:.3f} snapped {m['snapped_acc']:.3f} BLEU {m['bleu']:.1f} WER {m['wer']:.3f}"
            score = m["bleu"] + 100 * m["exact_acc"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(a), "itos": itos,
                            "train_sents": train_sents}, os.path.join(a.out, "best.pt"))
        print(msg)

    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    refs, hyps = predict(model, te, dev, a.mode, itos, train_sents)
    res = metrics(refs, hyps, train_sents)
    res.update({"split": a.split, "mode": a.mode, "n_test": len(refs)})
    print("\n=== TEST ===")
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    json.dump({"test": res, "val_history": history}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    with open(os.path.join(a.out, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["reference", "prediction"]); w.writerows(zip(refs, hyps))
    print("saved:", a.out)


if __name__ == "__main__":
    main()