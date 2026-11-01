"""Step 2: train + evaluate.

  python train.py --features features --split seen                # closed-set accuracy (100 known sentences)
  python train.py --features features --split unseen --n-unseen 10 # true translation test on sentences never trained on

Metrics reported on val every few epochs and on test at the end:
  exact_acc   : prediction == reference (after lowercase / punctuation strip)
  snapped_acc : prediction snapped to the closest TRAIN sentence, then exact match (closed-set accuracy)
  BLEU, WER
"""
import argparse, json, math, os, random, re, difflib, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sacrebleu

from data import read_manifest, make_splits, SignDataset, collate
from model import SLT


def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def snap(pred, train_sents):
    return max(train_sents, key=lambda s: difflib.SequenceMatcher(None, norm(pred), norm(s)).ratio())


def metrics(refs, hyps, train_sents):
    nr, nh = [norm(r) for r in refs], [norm(h) for h in hyps]
    exact = float(np.mean([a == b for a, b in zip(nr, nh)]))
    snapped = [norm(snap(h, train_sents)) for h in hyps]
    snap_acc = float(np.mean([a == b for a, b in zip(nr, snapped)]))
    bleu = sacrebleu.corpus_bleu(nh, [nr]).score
    w = float(np.mean([wer(a, b) for a, b in zip(nr, nh)]))
    return {"exact_acc": exact, "snapped_acc": snap_acc, "bleu": bleu, "wer": w}


@torch.no_grad()
def predict(model, loader, tok, dev, beams):
    model.eval()
    refs, hyps = [], []
    for kp, mask, sents in loader:
        hyps += model.translate(kp.to(dev), mask.to(dev), tok, num_beams=beams)
        refs += sents
    return refs, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--model", default="t5-small")
    ap.add_argument("--split", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--n-unseen", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup-epochs", type=int, default=8, help="epochs with T5 frozen (projector only)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--lr-t5", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/exp1")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    rows = read_manifest(os.path.join(a.features, "manifest.csv"))
    train, val, test = make_splits(rows, a.split, a.seed, n_unseen=a.n_unseen)
    train_sents = sorted({s for _, s in train})
    print(f"samples: train {len(train)} | val {len(val)} | test {len(test)} | train sentences {len(train_sents)}")
    if not val:
        raise SystemExit("Validation split is empty - need at least 3 videos per sentence.")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = SLT(a.model).to(dev)

    def tl(ds, shuffle):
        return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, collate_fn=collate, num_workers=0)
    tr_loader = tl(SignDataset(train, True), True)
    va_loader = tl(SignDataset(val, False), False)
    te_loader = tl(SignDataset(test, False), False)

    opt = torch.optim.AdamW([
        {"params": model.projector_params(), "lr": a.lr_proj},
        {"params": model.t5.parameters(), "lr": a.lr_t5},
    ], weight_decay=0.01)
    total = a.epochs * len(tr_loader)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()  # T5 is unstable in fp16

    best, history = -1, []
    for ep in range(1, a.epochs + 1):
        frozen = ep <= a.warmup_epochs
        for p in model.t5.parameters():
            p.requires_grad = not frozen
        model.train()
        run = 0.0
        for kp, mask, sents in tr_loader:
            lab = tok(sents, padding=True, truncation=True, max_length=64, return_tensors="pt").input_ids
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(kp.to(dev), mask.to(dev), lab.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
        msg = f"ep {ep:03d} loss {run/len(tr_loader):.4f}" + (" [T5 frozen]" if frozen else "")
        if ep % a.eval_every == 0 or ep == a.epochs:
            refs, hyps = predict(model, va_loader, tok, dev, beams=1)
            m = metrics(refs, hyps, train_sents)
            history.append({"epoch": ep, **m})
            msg += f" | val exact {m['exact_acc']:.3f} snapped {m['snapped_acc']:.3f} BLEU {m['bleu']:.1f} WER {m['wer']:.3f}"
            if m["bleu"] > best:
                best = m["bleu"]
                torch.save({"model": model.state_dict(), "args": vars(a), "train_sents": train_sents},
                           os.path.join(a.out, "best.pt"))
        print(msg)

    # ---- final test with best checkpoint ----
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    refs, hyps = predict(model, te_loader, tok, dev, beams=4)
    res = metrics(refs, hyps, train_sents)
    res.update({"split": a.split, "n_test": len(refs)})
    print("\n=== TEST ===")
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    json.dump({"test": res, "val_history": history}, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    with open(os.path.join(a.out, "test_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["reference", "prediction"]); w.writerows(zip(refs, hyps))
    print("saved:", a.out)


if __name__ == "__main__":
    main()