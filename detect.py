import os
import time
import math
import csv
import logging
import threading
import cv2
from ultralytics import YOLO

# ============================================================
#  WILDLIFE POACHING DETECTION SYSTEM
#  Uses YOLOv8 to detect persons + weapons in real-time
#  Triggers alerts when a weapon is detected near a person
# ============================================================

# -------------------- CONFIG --------------------------------
INPUT_SOURCE = 0          # 0 = webcam | r"C:\path\to\video.mp4"
LOOP_VIDEO   = False      # Repeat video file when it ends
MODEL        = "yolov8n.pt"

CONF_THRESH  = 0.12       # Minimum detection confidence
IMG_SIZE     = 720        # Inference image size (pixels)
NEAR_PIX     = 120        # Max pixel distance: weapon <-> person to trigger alert

SAVE_ALERT_FRAMES       = True
ALERT_PERSIST           = 3     # Consecutive frames needed to confirm alert
ALERT_COOLDOWN_SECONDS  = 5     # Seconds between successive alerts

OUTPUT_DIR = "alerts"
LOG_CSV    = "events.csv"

TARGET_INFER_FPS = 4.0    # Limit inference FPS for CPU friendliness
DEBUG_PRINT      = False
# ------------------------------------------------------------

# Weapon class names that YOLO might return
WEAPON_LABELS = {"knife", "gun", "pistol", "rifle", "scissors"}

# -------------------- LOGGING --------------------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG_PRINT else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PoachDemo")

# -------------------- STATE ----------------------------------
persist_count   = 0
last_alert_time = 0.0

# -------------------- SETUP ----------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "readable_time", "unix_time", "frame_id",
            "person_conf", "weapon_conf", "frame_file", "crop_file"
        ])

# ===================== HELPER FUNCTIONS ======================

def box_center(bb: tuple) -> tuple:
    """Return (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = bb
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def overlap_frac(a: tuple, b: tuple) -> float:
    """Fraction of box A that overlaps with box B."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter  = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area_a if area_a > 0 else 0.0


def iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two bounding boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter  = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list, thresh: float = 0.45) -> list:
    """Non-Maximum Suppression — removes duplicate detections."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    keep  = []
    for bb, conf in boxes:
        if all(iou(bb, kb) <= thresh for kb, _ in keep):
            keep.append((bb, conf))
    return keep


def draw_boxes(frame, persons: list, weapons: list, status: str):
    """Draw bounding boxes and status text onto the frame."""
    for (x1, y1, x2, y2), _ in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    for (x1, y1, x2, y2), _ in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Weapon", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    if status:
        color = (0, 0, 255) if "ALERT" in status else (0, 165, 255)
        cv2.putText(frame, status, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


# ==================== EVIDENCE SAVING ========================

def save_evidence(frame, frame_id: int, wbox: tuple,
                  pconf: float, wconf: float):
    """Save the alert frame + weapon crop and log to CSV."""
    tname      = time.strftime("%Y%m%d_%H%M%S")
    unix_ts    = int(time.time())

    frame_file = os.path.join(OUTPUT_DIR, f"alert_{frame_id}_{tname}.jpg")
    cv2.imwrite(frame_file, frame)

    x1, y1, x2, y2 = map(int, wbox)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

    crop_file = ""
    if x2 > x1 and y2 > y1:
        crop_file = os.path.join(OUTPUT_DIR, f"crop_{frame_id}_{tname}.jpg")
        cv2.imwrite(crop_file, frame[y1:y2, x1:x2])

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), unix_ts, frame_id,
            round(pconf, 3), round(wconf, 3),
            frame_file, crop_file
        ])

    log.warning("🚨 ALERT saved | frame=%s | p_conf=%.2f | w_conf=%.2f",
                frame_file, pconf, wconf)


# ==================== INFERENCE LOGIC ========================

def run_inference(model, frame, frame_id: int):
    """
    Run YOLO inference on a frame.
    Returns (persons, weapons, status_text).
    """
    global persist_count, last_alert_time

    result  = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
    persons = []
    weapons = []

    for bb, cls, conf in zip(result.boxes.xyxy,
                              result.boxes.cls,
                              result.boxes.conf):
        x1, y1, x2, y2 = map(int, bb.tolist())
        label = model.names[int(cls)]
        c     = float(conf)

        if label == "person":
            persons.append(((x1, y1, x2, y2), c))
        elif label in WEAPON_LABELS:
            weapons.append(((x1, y1, x2, y2), c))

    persons = nms(persons, 0.45)
    weapons = nms(weapons, 0.35)

    # --- Check if any weapon is near any person ---
    alert  = False
    best_w = None
    pconf  = wconf = 0.0

    for wbox, wc in weapons:
        wc_center = box_center(wbox)
        for pbox, pc in persons:
            pc_center = box_center(pbox)
            too_close = dist(wc_center, pc_center) < NEAR_PIX
            overlapping = overlap_frac(wbox, pbox) > 0.03
            if too_close or overlapping:
                alert  = True
                best_w = wbox
                pconf  = pc
                wconf  = wc
                break
        if alert:
            break

    # --- Persistence + cooldown logic ---
    persist_count = (persist_count + 1) if alert else 0

    now        = time.time()
    confirmed  = False
    status_text = ""

    if persist_count >= ALERT_PERSIST and (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS:
        confirmed      = True
        persist_count  = 0
        last_alert_time = now
        if SAVE_ALERT_FRAMES and best_w is not None:
            save_evidence(frame, frame_id, best_w, pconf, wconf)
        status_text = "🚨 ALERT: Weapon near person!"

    elif alert:
        status_text = f"⚠ Warning: possible weapon ({persist_count}/{ALERT_PERSIST})"

    if DEBUG_PRINT:
        log.debug("frame=%d | persons=%d | weapons=%d | persist=%d | status=%s",
                  frame_id, len(persons), len(weapons), persist_count, status_text or "clear")

    return persons, weapons, status_text


# ======================== MAIN ===============================

def run_file_mode(model):
    """Process a recorded video file (no threading needed)."""
    cap = cv2.VideoCapture(INPUT_SOURCE)
    if not cap.isOpened():
        log.error("Cannot open video file: %s", INPUT_SOURCE)
        return

    log.info("File mode | source: %s", INPUT_SOURCE)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if LOOP_VIDEO:
                cap.release()
                cap = cv2.VideoCapture(INPUT_SOURCE)
                continue
            log.info("Video finished.")
            break

        frame_id += 1
        persons, weapons, status = run_inference(model, frame, frame_id)
        draw_boxes(frame, persons, weapons, status)

        cv2.imshow("PoachDemo — File", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_webcam_mode(model):
    """Process live webcam feed using a background capture thread."""
    cap = cv2.VideoCapture(int(INPUT_SOURCE), cv2.CAP_DSHOW)
    if not cap.isOpened():
        log.error("Webcam not accessible (source=%s)", INPUT_SOURCE)
        return

    log.info("Webcam mode | source: %s | target FPS: %.1f",
             INPUT_SOURCE, TARGET_INFER_FPS)

    frame_lock = threading.Lock()
    shared     = {"frame": None, "stop": False}

    def cam_thread():
        while not shared["stop"]:
            ret, f = cap.read()
            if ret:
                with frame_lock:
                    shared["frame"] = f
            else:
                time.sleep(0.01)

    t = threading.Thread(target=cam_thread, daemon=True)
    t.start()

    frame_id        = 0
    last_infer_time = 0.0
    infer_interval  = 1.0 / max(0.1, TARGET_INFER_FPS)
    persons = weapons = []
    status  = ""

    while True:
        with frame_lock:
            src = shared["frame"]
        if src is None:
            time.sleep(0.01)
            continue

        frame     = src.copy()
        frame_id += 1
        now       = time.time()

        if now - last_infer_time >= infer_interval:
            last_infer_time = now
            persons, weapons, status = run_inference(model, frame, frame_id)

        draw_boxes(frame, persons, weapons, status)

        cv2.imshow("PoachDemo — Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    shared["stop"] = True
    t.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


def main():
    log.info("Loading model: %s", MODEL)
    model   = YOLO(MODEL)
    is_file = isinstance(INPUT_SOURCE, str) and os.path.exists(INPUT_SOURCE)

    if is_file:
        run_file_mode(model)
    else:
        run_webcam_mode(model)


if __name__ == "__main__":
    main()