import cv2

def put_tag(img, text, x=10, y=30):
    cv2.rectangle(img, (x-8, y-24), (x+8+8*len(text), y+8), (0,0,0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def draw_fps(img, fps):
    cv2.putText(img, f"{fps:.1f} FPS", (img.shape[1]-120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
