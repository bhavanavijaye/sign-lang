import csv
p = "features2/manifest.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8")))
for r in rows:
    if r["sentence"].strip().lower() == "i dont agree":
        r["sentence"] = "i do not agree"
with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["feature_path", "sentence", "source"])
    w.writeheader(); w.writerows(rows)
print("unique sentences:", len({r["sentence"] for r in rows}))