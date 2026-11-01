"""Step 1: videos / frame-folders -> keypoint .npy files + manifest.csv

Usage:
  python extract_keypoints.py --data D:/isl_dataset --inspect          # look at structure only
  python extract_keypoints.py --data D:/isl_dataset --out features
  # if labels live in a CSV (video name -> sentence):
  python extract_keypoints.py --data D:/isl_dataset --csv labels.csv --csv-key video --csv-text sentence

Label rule when no CSV is given: the folder name the sample lives in
(e.g. .../how are you/vid1.mp4 -> "how are you").
Check manifest.csv afterwards and fix labels there if they look wrong.
"""
import argparse, os, re, csv, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2

from keypoints import frame_to_raw, normalize

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def find_samples(root: Path):
    """Return list of (kind, path). kind in {'video','frames'}."""
    samples = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        vids = [f for f in filenames if Path(f).suffix.lower() in VIDEO_EXT]
        imgs = [f for f in filenames if Path(f).suffix.lower() in IMG_EXT]
        for v in vids:
            samples.append(("video", d / v))
        if imgs and not vids:
            samples.append(("frames", d))
    samples.sort(key=lambda s: natural_key(s[1]))
    return samples


def clean_label(name):
    name = Path(name).stem if Path(name).suffix.lower() in VIDEO_EXT else name
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_csv_map(path, key_col, text_col):
    m = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = Path(str(row[key_col])).stem.strip().lower()
            m[k] = str(row[text_col]).strip()
    return m


def iter_frames(kind, path, stride, max_frames):
    """Yield RGB frames one at a time (low memory: never holds a whole video in RAM)."""
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0:
            keep = list(range(0, n, stride))
            if len(keep) > max_frames:
                pick = np.linspace(0, len(keep) - 1, max_frames).round().astype(int)
                keep = [keep[j] for j in pick]
            keep = set(keep)
            i = 0
            while i <= max(keep):
                if i in keep:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                elif not cap.grab():
                    break
                i += 1
        else:  # frame count unknown: plain stride
            i = cnt = 0
            while cnt < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    cnt += 1
                i += 1
        cap.release()
    else:
        files = sorted([p for p in Path(path).iterdir() if p.suffix.lower() in IMG_EXT], key=natural_key)[::stride]
        if len(files) > max_frames:
            pick = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
            files = [files[j] for j in pick]
        for p in files:
            fr = cv2.imread(str(p))
            if fr is not None:
                yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


_holistic = None


def _init(complexity=1, conf=0.5, static=False):
    global _holistic
    import mediapipe as mp
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=static, model_complexity=complexity, refine_face_landmarks=False,
        min_detection_confidence=conf, min_tracking_confidence=conf)


def _work(args):
    idx, kind, path, out_dir, stride, max_frames = args
    out_path = Path(out_dir) / f"{idx:05d}.npy"
    if out_path.exists():
        return idx, str(out_path), True, None
    res = [frame_to_raw(_holistic, f) for f in iter_frames(kind, path, stride, max_frames)]
    if len(res) < 4:
        return idx, None, False, None
    raws = [r for r, _ in res]
    ratio = float(np.mean([h for _, h in res]))  # fraction of frames with at least one hand
    feats = normalize(np.stack(raws))
    np.save(out_path, feats)
    return idx, str(out_path), True, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="features")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--csv"); ap.add_argument("--csv-key", default="video"); ap.add_argument("--csv-text", default="sentence")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe model complexity")
    ap.add_argument("--conf", type=float, default=0.5, help="detection/tracking confidence (lower = more hands found)")
    ap.add_argument("--static", action="store_true", help="re-detect hands on every frame (slower, finds more)")
    ap.add_argument("--limit", type=int, default=0, help="only process N evenly spaced videos (quick settings test)")
    a = ap.parse_args()

    root = Path(a.data)
    samples = find_samples(root)
    print(f"found {len(samples)} samples ({sum(k=='video' for k,_ in samples)} videos, "
          f"{sum(k=='frames' for k,_ in samples)} frame-folders)")
    if not samples:
        sys.exit("No videos or image folders found. Run with --inspect and check the path.")

    csv_map = load_csv_map(a.csv, a.csv_key, a.csv_text) if a.csv else None
    labels, missing = [], 0
    for kind, p in samples:
        if csv_map is not None:
            key = (p.stem if kind == "video" else p.name).lower()
            lab = csv_map.get(key)
            if lab is None:
                missing += 1
                lab = ""
        else:
            lab = clean_label(p.parent.name if kind == "video" else p.name)
        labels.append(lab)
    if csv_map is not None:
        print(f"CSV matched {len(samples)-missing}/{len(samples)} samples")
    print(f"unique labels: {len(set(labels))}")
    for (k, p), l in list(zip(samples, labels))[:8]:
        print(f"  {p}  ->  '{l}'")
    if a.inspect:
        return

    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, k, p, a.out, a.stride, a.max_frames) for i, (k, p) in enumerate(samples) if labels[i]]
    if a.limit and a.limit < len(jobs):
        step = len(jobs) / a.limit
        jobs = [jobs[int(i * step)] for i in range(a.limit)]
    rows, ratios = [], []
    with Pool(a.workers, initializer=_init, initargs=(a.complexity, a.conf, a.static)) as pool:
        for n, (idx, path, ok, ratio) in enumerate(pool.imap_unordered(_work, jobs), 1):
            if ratio is not None:
                ratios.append(ratio)
            if ok:
                rows.append((path, labels[idx], str(samples[idx][1])))
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", end="\r")
    rows.sort()
    with open(Path(a.out) / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature_path", "sentence", "source"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} samples -> {a.out}/manifest.csv")
    if ratios:
        print(f"frames with at least one hand detected: {np.mean(ratios):.3f}  "
              f"(settings: complexity={a.complexity}, conf={a.conf}, static={a.static}, stride={a.stride})")


if __name__ == "__main__":
    main()