"""Print every run in runs/ as a table, plus mean +/- std for seed groups.

  python collect_results.py            # looks in runs/
  python collect_results.py --csv results_table.csv

Runs named like cls_f2_s1, cls_f2_s2, cls_f2_s3 are grouped as 'cls_f2' (the _s<N> suffix is the seed).
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict
from statistics import mean, pstdev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(f))
        t = json.load(open(f))["test"]
        rows.append({"run": name, "split": t.get("split", ""), "mode": t.get("mode", "t5"),
                     "exact_acc": t["exact_acc"] * 100, "bleu": t["bleu"], "wer": t["wer"],
                     "n_test": t.get("n_test", "")})
    if not rows:
        raise SystemExit(f"No results.json found under {a.runs}/")

    print(f"\n{'run':28s}{'split':8s}{'mode':10s}{'exact%':>8s}{'BLEU':>8s}{'WER':>7s}")
    for r in rows:
        print(f"{r['run']:28s}{r['split']:8s}{r['mode']:10s}{r['exact_acc']:8.1f}{r['bleu']:8.1f}{r['wer']:7.3f}")

    groups = defaultdict(list)
    for r in rows:
        groups[re.sub(r"_s\d+$", "", r["run"])].append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print(f"\n{'group (mean +/- std over seeds)':32s}{'runs':>5s}{'exact%':>16s}{'BLEU':>16s}")
        for k, v in multi.items():
            e, b = [x["exact_acc"] for x in v], [x["bleu"] for x in v]
            print(f"{k:32s}{len(v):5d}{mean(e):9.1f} +/- {pstdev(e):4.1f}{mean(b):9.1f} +/- {pstdev(b):4.1f}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("\nsaved", a.csv)


if __name__ == "__main__":
    main()