# test_video_read.py
import cv2, os, time
PATH = r"C:\Users\MUDIT BHARDWAJ\Videos\sample.mp4"   # change if needed

print("Exists:", os.path.exists(PATH))
cap = cv2.VideoCapture(PATH)   # don't force CAP_DSHOW for files
print("Opened:", cap.isOpened())
print("Backend:", cap.getBackendName() if hasattr(cap, 'getBackendName') else "unknown")
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Width x Height:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_id = 0
fail_count = 0
start = time.time()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            print(f"[READ FAIL] frame_id={frame_id} fail_count={fail_count}")
            # if many consecutive fails -> assume EOF or unrecoverable
            if fail_count >= 5:
                print("Multiple consecutive read failures -> stopping.")
                break
            time.sleep(0.05)
            continue
        fail_count = 0
        frame_id += 1
        if frame_id % 100 == 0:
            print("Read frame", frame_id, "time elapsed:", round(time.time()-start,1))
        # very small work to avoid hogging
        if frame_id >= 5000:
            print("Reached 5000 frames, stopping.")
            break
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    cap.release()
    print("Finished. frames read:", frame_id, " elapsed:", round(time.time()-start,1))
