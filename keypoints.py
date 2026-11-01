"""MediaPipe Holistic -> normalized (75, 3) keypoints per frame.

75 points = 33 pose + 21 left hand + 21 right hand.
Missing landmarks are kept as exact zeros.
"""
import numpy as np

N_POSE, N_HAND = 33, 21
N_PTS = N_POSE + 2 * N_HAND  # 75


def _lm_to_arr(lms, n):
    if lms is None:
        return np.zeros((n, 3), dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)


def frame_to_raw(holistic, frame_rgb):
    """Run MediaPipe on one RGB frame. Returns (75, 3) raw array and hands_present flag."""
    res = holistic.process(frame_rgb)
    pose = _lm_to_arr(res.pose_landmarks, N_POSE)
    lh = _lm_to_arr(res.left_hand_landmarks, N_HAND)
    rh = _lm_to_arr(res.right_hand_landmarks, N_HAND)
    hands = res.left_hand_landmarks is not None or res.right_hand_landmarks is not None
    return np.concatenate([pose, lh, rh], 0), hands


def normalize(raw):
    """raw: (T, 75, 3). Center on mid-shoulder, scale by shoulder width. Missing stays 0."""
    x = raw.copy()
    missing = (x == 0).all(-1)  # (T, 75)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        if missing[t, 11] or missing[t, 12]:  # no shoulders -> cannot normalize
            continue
        c = (x[t, 11] + x[t, 12]) / 2
        s = np.linalg.norm(x[t, 11, :2] - x[t, 12, :2])
        s = s if s > 1e-4 else 1.0
        out[t] = (x[t] - c) / s
        out[t][missing[t]] = 0.0
    return out.astype(np.float32)
    