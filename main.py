import cv2, time, json
import numpy as np
import mediapipe as mp
from gestures import classify_gesture
from utils.draw import put_tag, draw_fps
from tts import Speaker
from collections import deque

# --- Load labels mapping ---
with open("assets/labels.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def to_np(landmark_list, w, h):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    speaker = Speaker(rate=180)
    last_spoken = ""
    stable_label = None
    stable_count = 0
    speak_cooldown = 0  # frames
    t0 = time.time(); frames = 0

    wrist_hist = deque(maxlen=20)  # for NO gesture (wave)

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label = None

            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                handedness = res.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                lm = to_np(hand_landmarks.landmark, w, h)

                # history for NO gesture (wrist x normalized 0..1)
                wrist_hist.append(lm[0][0])

                label = classify_gesture(lm, handedness, list(wrist_hist))

            # --- Stability logic: avoid jitter & spam ---
            if label == stable_label and label is not None:
                stable_count += 1
            else:
                stable_label = label
                stable_count = 0

            if speak_cooldown > 0:
                speak_cooldown -= 1

            phrase = None
            if stable_label and stable_count > 6 and speak_cooldown == 0:
                phrase = LABELS.get(stable_label, stable_label.title())
                # speak only if changed
                if phrase != last_spoken:
                    try:
                        speaker.say(phrase)
                        last_spoken = phrase
                        speak_cooldown = 30  # ~1s at 30 fps
                    except Exception as e:
                        print("TTS error:", e)

            # --- UI overlays ---
            put_tag(frame, f"Gesture: {stable_label or '—'}")
            if phrase:
                put_tag(frame, f"Speak: {phrase}", x=10, y=60)

            # fps
            frames += 1
            if frames % 10 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
            draw_fps(frame, fps if frames >= 10 else 0)

            cv2.imshow("SignLink – Real-Time Sign to Voice", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
