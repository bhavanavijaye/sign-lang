"""7-fold cross-validation (Bi-LSTM classifier).

Every video is tested exactly once, and each fold trains on about 6 videos per sentence
(the fixed split trains on about 5). There is NO validation set and NO checkpoint picking:
the model after the last epoch is evaluated.

  python crossval.py --features features2 --folds 7 --out runs\\cv_cls

Takes about 7 x (one classifier run), roughly an hour. Results: runs\\cv_cls\\results.json
(readable by collect_results.py) and cv_predictions.csv.
"""
import argparse, csv, json, math, os, random
from collections import defaultdict
from statistics import mean, pstdev
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from data import read_manifest, SignDataset, collate
from model_bilstm import BiLSTMClassifier
from train import metrics, norm


def make_folds(rows, k, seed):
    """Assign every video to a fold, spreading each sentence's videos over different folds.
    Sentences with fewer than 2 videos cannot be tested fairly and are skipped."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    fold_of, skipped = {}, []
    for s, items in sorted(by.items()):
        if len(items) < 2:
            skipped.append(s)
            continue
        items = items[:]
        rng.shuffle(items)
        off = rng.randrange(k)
        for j, r in enumerate(items):
            fold_of[r[0]] = (j + off) % k
    return fold_of, skipped


def run_fold(train_rows, test_rows, a, dev, fold):
    train_sents = sorted({s for _, s in train_rows})
    s2c = {s: i for i, s in enumerate(train_sents)}
    torch.manual_seed(a.seed + fold)
    model = BiLSTMClassifier(len(train_sents), hid=a.hid).to(dev)
    tr = DataLoader(SignDataset(train_rows, True), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=0)
    te = DataLoader(SignDataset(test_rows, False), batch_size=a.bs, shuffle=False, collate_fn=collate, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(tr)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        for kp, mask, sents in tr:
            y = torch.tensor([s2c[s] for s in sents], device=dev)
            loss = ce(model(kp.to(dev), mask.to(dev)), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
        if ep % 25 == 0 or ep == a.epochs:
            print(f"  fold {fold} ep {ep:03d} loss {run / len(tr):.4f}", flush=True)
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for kp, mask, sents in te:
            idx = model(kp.to(dev), mask.to(dev)).argmax(-1).cpu().tolist()
            hyps += [train_sents[i] for i in idx]
            refs += sents
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features2")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/cv_cls")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = read_manifest(os.path.join(a.features, "manifest.csv"))
    fold_of, skipped = make_folds(rows, a.folds, a.seed)
    if skipped:
        print("skipped (fewer than 2 videos):", skipped)
    all_sents = sorted({s for _, s in rows})
    all_refs, all_hyps, accs = [], [], []
    for k in range(a.folds):
        test_rows = [r for r in rows if fold_of.get(r[0]) == k]
        train_rows = [r for r in rows if r[0] in fold_of and fold_of[r[0]] != k]
        print(f"fold {k + 1}/{a.folds}: train {len(train_rows)} | test {len(test_rows)}", flush=True)
        refs, hyps = run_fold(train_rows, test_rows, a, dev, k)
        acc = float(np.mean([norm(x) == norm(y) for x, y in zip(refs, hyps)]))
        accs.append(acc)
        print(f"fold {k + 1} exact accuracy: {acc * 100:.1f}%", flush=True)
        all_refs += refs; all_hyps += hyps

    res = metrics(all_refs, all_hyps, all_sents)
    res.update({"split": f"cv{a.folds}", "mode": "classify", "n_test": len(all_refs)})
    print("\n=== CROSS-VALIDATION ===")
    print("per-fold accuracy %:", [round(x * 100, 1) for x in accs])
    print(f"mean {mean(accs) * 100:.1f}% +/- {pstdev(accs) * 100:.1f}  | pooled exact_acc {res['exact_acc'] * 100:.1f}%  "
          f"BLEU {res['bleu']:.1f}  WER {res['wer']:.3f}  ({res['n_test']} videos)")
    json.dump({"test": res, "fold_acc": accs}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    with open(os.path.join(a.out, "cv_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["reference", "prediction"]); w.writerows(zip(all_refs, all_hyps))
    print("saved:", a.out)


if __name__ == "__main__":
    main()