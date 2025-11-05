import numpy as np

TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
PIP_IDS = [3, 6, 10, 14, 18]   # joints below tips

def _is_finger_up(lm, tip, pip, axis='y'):
    # For non-thumb: tip is "higher" (smaller y) than its PIP when palm faces camera.
    return lm[tip][1] < lm[pip][1] if axis == 'y' else lm[tip][0] > lm[pip][0]

def _thumb_extended(lm, right_hand=True):
    # Thumb logic: for right hand, extended if tip.x < ip.x (points left); left hand opposite.
    tip_x = lm[4][0]; ip_x = lm[3][0]
    return tip_x < ip_x if right_hand else tip_x > ip_x

def fingers_state(landmarks, handeness_label):
    """
    landmarks: np.array shape (21,3) normalized to image coords (x,y)
    handeness_label: 'Right' or 'Left' from MediaPipe
    returns dict of booleans
    """
    right = (handeness_label == 'Right')
    up = {}
    # Thumb
    up['thumb']  = _thumb_extended(landmarks, right_hand=right)
    # Other fingers (compare tip.y < pip.y)
    up['index']  = _is_finger_up(landmarks, 8, 6, 'y')
    up['middle'] = _is_finger_up(landmarks, 12,10,'y')
    up['ring']   = _is_finger_up(landmarks, 16,14,'y')
    up['pinky']  = _is_finger_up(landmarks, 20,18,'y')
    return up

def distance(p1, p2):
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def classify_gesture(landmarks, hand_label, history=None):
    """
    Returns one of: OPEN_PALM, FIST, OK, THUMBS_UP, THUMBS_DOWN, PEACE, STOP, NO, None
    history: optional list of previous wrist x positions to detect waving (NO)
    """
    lm = landmarks  # (21,3)
    state = fingers_state(lm, hand_label)
    up_count = sum(state.values())

    wrist = lm[0]; index_tip = lm[8]; middle_tip = lm[12]; thumb_tip = lm[4]
    index_pip = lm[6]; middle_pip = lm[10]

    # Simple, reliable classes:

    # 1) OPEN PALM / STOP (all up)
    if up_count >= 4 and state['index'] and state['middle'] and state['ring'] and state['pinky']:
        # if hand is front-facing and static = STOP; otherwise generic open palm = HELLO
        return "OPEN_PALM"

    # 2) FIST (none up)
    if up_count == 0:
        return "FIST"

    # 3) THUMBS UP / DOWN (thumb only up & vertical orientation)
    vertical_delta = index_pip[1] - wrist[1]
    thumb_only = state['thumb'] and not(state['index'] or state['middle'] or state['ring'] or state['pinky'])
    if thumb_only:
        if vertical_delta > 0:  # fingers below wrist (hand upright)
            # decide up vs down by thumb tip above/below wrist
            return "THUMBS_UP" if thumb_tip[1] < wrist[1] else "THUMBS_DOWN"

    # 4) OK (thumb tip close to index tip, others down/neutral)
    if distance(thumb_tip, index_tip) < 0.05 and not state['middle']:
        return "OK"

    # 5) PEACE (index & middle up, others down)
    if state['index'] and state['middle'] and not(state['ring'] or state['pinky']) and not state['thumb']:
        return "PEACE"

    # 6) NO (index waving left-right) – detect oscillation with history
    if history is not None and len(history) >= 6 and state['index'] and not state['middle']:
        # if oscillation amplitude exceeds threshold, call it NO
        amp = max(history) - min(history)
        if amp > 0.06:
            return "NO"

    return None
