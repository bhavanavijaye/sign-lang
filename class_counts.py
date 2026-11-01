"""Do sentences with fewer videos get more wrong?"""
import csv, glob, re
from collections import Counter


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


rows = list(csv.DictReader(open("features2/manifest.csv", encoding="utf-8")))
count = Counter(norm(r["sentence"]) for r in rows)

total, wrong = Counter(), Counter()
files = sorted(glob.glob("runs/cls_f2_s*/test_predictions.csv"))
for f in files:
    for r in csv.DictReader(open(f, encoding="utf-8")):
        ref = norm(r["reference"])
        total[ref] += 1
        wrong[ref] += ref != norm(r["prediction"])

print(f"{len(count)} sentences, {sum(count.values())} videos, {len(files)} classifier runs\n")
print("videos per sentence (how many sentences have that many):")
for k, v in sorted(Counter(count.values()).items()):
    print(f"  {k:2d} videos: {v} sentences")

buckets = [("3 to 5 videos", 0, 5), ("6 videos", 6, 6), ("7 videos", 7, 7), ("8 or more", 8, 999)]
print("\nclassifier accuracy by how many videos the sentence has:")
for name, lo, hi in buckets:
    sents = [s for s, c in count.items() if lo <= c <= hi]
    t = sum(total[s] for s in sents)
    w = sum(wrong[s] for s in sents)
    if t:
        print(f"  {name:14s} {len(sents):3d} sentences, {t:3d} test videos, accuracy {100 * (t - w) / t:5.1f}%")

print("\nsentences with the fewest videos:")
for s, c in sorted(count.items(), key=lambda x: x[1])[:8]:
    print(f"  {c} videos  {s}   (wrong {wrong[s]}/{total[s]})")