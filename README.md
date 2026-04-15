# 🦁 Wildlife Poaching Detection System

An AI-powered surveillance system that detects potential wildlife poaching activity by identifying humans carrying weapons in real-time video or webcam feeds.

This project uses YOLOv8 for object detection and applies custom logic to trigger alerts when a weapon is detected near a person.

---

## 🚀 Features

- 🎥 Real-time detection using webcam or video
- 🧠 AI-based object detection using YOLOv8
- 🔫 Detects weapons like knife, gun, pistol, rifle, etc.
- 🚨 Smart alert system (weapon near person)
- 📸 Saves alert frames as evidence
- 📝 Logs events into CSV file
- ⚡ Optimized for CPU performance

---

## 🛠️ Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy

---

## 📂 Project Structure

wildlife-poaching-detection-ml/
│
├── detect.py              # Main detection script
├── yolov8n.pt            # YOLO model (auto-download recommended)
├── alerts/               # Saved alert images
├── events.csv            # Detection logs
├── README.md
└── .venv/

---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/MuditBh/wildlife-poaching-detection-ml.git  
cd wildlife-poaching-detection-ml

---

### 2. Create virtual environment
python -m venv .venv  
.venv\Scripts\activate  

---

### 3. Install dependencies
pip install ultralytics opencv-python  

---

### 4. Run the project
python detect.py  

---

## 🎥 Camera Setup (Important)

If the webcam is not working or shows an error like:

Webcam not accessible (source=0)

This usually means the default camera index is incorrect.

### 🔧 Fix

Try changing the camera index in `detect.py`:

```python
INPUT_SOURCE = 1 or INPUT_SOURCE = 2

```

### Find correct camera index
```python
for i in range(5):
    cap = cv2.VideoCapture(i)
    print(f"Index {i}:", cap.isOpened())
    cap.release()

```

## 🎯 How it works

- YOLOv8 detects objects in each frame  
- Filters persons and weapons  
- Calculates distance between them  
- If a weapon is close to a person → alert triggered  
- Saves frame and logs event  

---

## 📸 Output

- Bounding boxes on detected objects  
- Alert message on screen  
- Saved evidence in alerts/  
- Logs stored in events.csv  

---

## ⚠️ Note

- Webcam may not work on some systems due to permissions  
- For best results, use a video file as input  

---

## 🔮 Future Improvements

- Custom-trained weapon detection model  
- Email/SMS alert system  
- Web dashboard integration  
- Deployment on edge devices (Raspberry Pi / CCTV)  

---

## 👨‍💻 Author

Mudit Bhardwaj  

---

## ⭐ Support

If you like this project, consider giving it a star ⭐ on GitHub!
