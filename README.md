# 🛣️ Road Damage Detection (`YOLO26m`)

An interactive **Streamlit** application for detecting and classifying road surface damages—**crack**, **alligator**, **block**, and **pothole**—using a trained **YOLO26m** object detection model.

Developed as **Project #2** in **DataSpace #2** as part of the **Artificial Intelligence Diploma** with **Instant**, under the supervision of **Eng. Jehan** and **Eng. Sohila**.

---

## ✨ Key Features

- **Real-Time Inference UI:** Upload road images (`JPG`, `JPEG`, `PNG`) and inspect side-by-side original and annotated outputs.
- **Interactive Controls:** Adjust `Confidence Threshold (conf)`, `IoU Threshold (iou)`, `Image Size (imgsz)`, and toggle `Test-Time Augmentation (TTA)`.
- **Class Breakdown & Summary:** Displays total detections, unique damage types found, peak confidence, and per-class detection counts.
- **Automatic Weights Download:** Automatically fetches `best.pt` from **GitHub Releases** (`v1.0`) if not already present locally.
- **Export Annotated Result:** Download the predicted image directly as a `PNG`.

---

## 🔍 Model Comparison & Selection

During experimentation, three YOLO generations were evaluated on road damage classes (`crack`, `alligator`, `block`, `pothole`):

| Model | Role in Evaluation | Notes |
| :--- | :--- | :--- |
| **YOLOv8** | Baseline comparison | Evaluated across road surface classes for detection behavior and localization |
| **YOLOv11** | Intermediate comparison | Compared against YOLOv8 on multi-class road damage patterns |
| **YOLO26m** | **Selected Model** | Chosen as the strongest overall fit for deployment in the Streamlit inspector |

---

## 📁 Project Structure

```text
road-damage-detection/
├── app.py              # Streamlit application & automatic weights loader
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignored local artifacts & large weight files
└── README.md           # Project documentation
```

---

## 🚀 Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mhmdsamhsyn09-wq/road-damage-detection.git](https://github.com/mhmdsamhsyn09-wq/road-damage-detection.git)
   cd road-damage-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *(On first run without a local `best.pt`, the app downloads the weights automatically from the `v1.0` release.)*

---

## 🙏 Acknowledgments

Special thanks to the **Instant** team and my training supervisors, **Eng. Jehan** and **Eng. Sohila**, for their continuous guidance, feedback, and support throughout **DataSpace #2**.
