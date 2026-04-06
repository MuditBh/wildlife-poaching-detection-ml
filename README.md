# 🦁 Wildlife Poaching Detection System

A real-time AI-powered surveillance system that detects potential poaching activity by identifying **persons carrying weapons** in live webcam feeds or recorded video footage.

Built with **YOLOv8** (Ultralytics) and **OpenCV**.

---

## 🎯 How It Works

1. Each frame is analyzed by YOLOv8 to detect `person` and weapon classes (`knife`, `gun`, `pistol`, `rifle`, `scissors`)
2. If a weapon is detected **near or overlapping** a person, an alert is triggered
3. The alert requires **3 consecutive confirmations** (configurable) to avoid false positives
4. Confirmed alerts are **saved as images** and **logged to CSV**

---

## 📸 Demo

| Normal Frame | Alert Triggered |
|---|---|
| Green box = Person | Red box = Weapon |
| No warning shown | `🚨 ALERT: Weapon near person!` |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/poach-demo.git
cd poach-demo
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> Skip this if you already have the packages installed.
```

### 4. Download YOLOv8 model
```bash
# Automatically downloaded on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 5. Run
```bash
python detect.py
```

Press **`Q`** to quit the window.

---

## ⚙️ Configuration

Edit the `CONFIG` block at the top of `detect.py`:

| Variable | Default | Description |
|---|---|---|
| `INPUT_SOURCE` | `0` | `0` = webcam, or path to video file |
| `LOOP_VIDEO` | `False` | Repeat video file when finished |
| `MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `CONF_THRESH` | `0.12` | Minimum detection confidence |
| `NEAR_PIX` | `120` | Max pixel distance weapon↔person |
| `ALERT_PERSIST` | `3` | Frames needed to confirm alert |
| `ALERT_COOLDOWN_SECONDS` | `5` | Seconds between alerts |
| `SAVE_ALERT_FRAMES` | `True` | Save alert images to disk |
| `TARGET_INFER_FPS` | `4.0` | Inference rate (CPU-friendly) |

### Using a video file:
```python
INPUT_SOURCE = r"C:\path\to\video.mp4"
LOOP_VIDEO   = True
```

---

## 📁 Project Structure

```
poach-demo/
├── detect.py          # Main detection script
├── requirements.txt   # Python dependencies
├── README.md
├── .gitignore
└── alerts/            # Auto-created: saved alert frames & crops
    ├── alert_42_20240101_120000.jpg
    └── crop_42_20240101_120000.jpg
events.csv             # Auto-created: alert log
```

---

## 📊 Alert Log (events.csv)

Each confirmed alert is logged with:

| Column | Description |
|---|---|
| `readable_time` | Human-readable timestamp |
| `unix_time` | Unix timestamp |
| `frame_id` | Frame number |
| `person_conf` | Person detection confidence |
| `weapon_conf` | Weapon detection confidence |
| `frame_file` | Path to saved alert frame |
| `crop_file` | Path to weapon crop image |

---

## 🧠 Model

Uses **YOLOv8n** (nano) by default — fast enough for real-time on CPU.

For better accuracy, switch to a larger model:
```python
MODEL = "yolov8s.pt"   # small
MODEL = "yolov8m.pt"   # medium
```

> **Note:** `yolov8n.pt` is auto-downloaded by Ultralytics and is excluded from this repo via `.gitignore` due to file size.

---

## 📦 Requirements

- Python 3.8+
- Webcam or video file
- No GPU required (CPU inference supported)

---

## 👨‍💻 Author

**Naman Goel**  
B.Tech CSE | GNIOT, Greater Noida  
[GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — free to use and modify.
