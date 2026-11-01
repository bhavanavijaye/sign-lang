"""Step 3: live webcam translation.

  python infer_live.py --ckpt runs/exp1/best.pt

Logic: collect frames while hands are visible; when hands drop out of view for ~0.7s
(a pause between sentences) translate the collected clip and show the English text.
Also force a translation if a clip runs longer than --max-seconds.
Press q to quit.
"""
import argparse, time, collections
import numpy as np
import torch, cv2
import mediapipe as mp
from transformers import AutoTokenizer

from keypoints import frame_to_raw, normalize
from data import to_features, resample, fill_gaps, FILL_MAX_GAP, EMB_DIR, MAX_T
from model import SLT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.7, help="seconds without hands = sentence end")
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--stride", type=int, default=2, help="must match extraction stride")
    a = ap.parse_args()
    if EMB_DIR:
        raise SystemExit("Live demo for RGB-embedding models is not built yet. Clear it with: Remove-Item Env:ISL_EMB_DIR")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    name = ck["args"]["model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = SLT(name).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    holistic = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(a.cam)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    eff_fps = fps / a.stride
    raws, no_hand, i = [], 0, 0
    history = collections.deque(maxlen=4)

    def translate(clip):
        x = normalize(np.stack(clip))
        if FILL_MAX_GAP:
            x = fill_gaps(x, FILL_MAX_GAP)
        if len(x) > MAX_T:
            x = resample(x, MAX_T)
        kp = torch.from_numpy(to_features(x))[None].to(dev)
        mask = torch.ones(1, kp.shape[1], dtype=torch.long, device=dev)
        return model.translate(kp, mask, tok, num_beams=4)[0]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % a.stride == 0:
            raw, hands = frame_to_raw(holistic, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if hands:
                raws.append(raw); no_hand = 0
            elif raws:
                raws.append(raw); no_hand += 1
            ended = raws and no_hand / eff_fps >= a.pause
            too_long = len(raws) / eff_fps >= a.max_seconds
            if (ended or too_long) and len(raws) - no_hand >= 8:
                clip = raws[: len(raws) - no_hand] if ended else raws
                history.append(translate(clip))
                raws, no_hand = [], 0
            elif ended:
                raws, no_hand = [], 0
        y = 30
        for line in history:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 32
        if raws:
            cv2.putText(frame, "signing...", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("ISL -> English", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()