from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage Detection", layout="wide")

WEIGHTS_PATH = "best.pt"
IMGSZ = 640


@st.cache_resource
def load_model(weights_path: str):
  return YOLO(weights_path)


st.title("Road Damage Detection (YOLO26m)")
st.caption(
    "Classes: crack, alligator, block, pothole | Default imgsz=640"
)

if not Path(WEIGHTS_PATH).exists():
  st.error(f"ملف الأوزان {WEIGHTS_PATH} غير موجود بجانب app.py")
  st.stop()

model = load_model(WEIGHTS_PATH)

st.sidebar.header("Inference Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.05, 0.90, 0.25, 0.05)
iou = st.sidebar.slider("IoU Threshold", 0.10, 0.90, 0.50, 0.05)
use_tta = st.sidebar.checkbox("Enable TTA (augment=True)", value=True)

uploaded_file = st.file_uploader(
    "ارفع صورة طريق للفحص", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  img_np = np.array(image)

  results = model.predict(
      source=img_np,
      imgsz=IMGSZ,
      conf=conf,
      iou=iou,
      augment=use_tta,
      verbose=False,
  )

  annotated_bgr = results[0].plot()
  annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
  with col2:
    st.subheader("Detected Damages")
    st.image(annotated_rgb, use_container_width=True)

  boxes = results[0].boxes
  st.subheader(f"Detections Count: {len(boxes)}")
  if len(boxes) > 0:
    rows = []
    for box in boxes:
      cls_id = int(box.cls[0].item())
      cls_name = model.names[cls_id]
      score = float(box.conf[0].item())
      rows.append({"Class": cls_name, "Confidence": round(score, 3)})
    st.dataframe(rows, use_container_width=True)