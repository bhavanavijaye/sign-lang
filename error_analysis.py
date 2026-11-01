"""Which sentences does the model get wrong, and what does it confuse them with?
Reads runs/<group>_s*/test_predictions.csv for the seen-split groups."""
import csv, glob, re
from collections import Counter

GROUPS = ["cls_f2", "seq_f2", "t5_f2"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


for g in GROUPS:
    files = sorted(glob.glob(f"runs/{g}_s*/test_predictions.csv"))
    if not files:
        continue
    total, wrong, conf = Counter(), Counter(), Counter()
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            ref, hyp = norm(r["reference"]), norm(r["prediction"])
            total[ref] += 1
            if ref != hyp:
                wrong[ref] += 1
                conf[(ref, hyp)] += 1
    n, w = sum(total.values()), sum(wrong.values())
    print(f"\n##### {g}: {len(files)} runs, {w}/{n} wrong")
    print("sentences with most errors (wrong / appearances):")
    for s, c in wrong.most_common(10):
        print(f"  {c:2d} / {total[s]:2d}  {s}")
    print("most common confusions (reference -> prediction):")
    for (a, b), c in conf.most_common(8):
        print(f"  {c:2d}x  {a}  ->  {b}")