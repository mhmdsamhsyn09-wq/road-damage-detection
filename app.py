from collections import Counter
import io
from pathlib import Path
import urllib.request

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Road Damage Detection | YOLO26m",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

WEIGHTS_PATH = Path("best.pt")
WEIGHTS_URL = "https://github.com/mhmdsamhsyn09-wq/road-damage-detection/releases/download/v1.0/best.pt"

CLASS_META = {
    "crack": {
        "label": "Crack",
        "color": "#38BDF8",
        "bg": "rgba(56, 189, 248, 0.14)",
        "border": "rgba(56, 189, 248, 0.45)",
    },
    "alligator": {
        "label": "Alligator Crack",
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.14)",
        "border": "rgba(245, 158, 11, 0.45)",
    },
    "block": {
        "label": "Block Crack",
        "color": "#A78BFA",
        "bg": "rgba(167, 139, 250, 0.14)",
        "border": "rgba(167, 139, 250, 0.45)",
    },
    "pothole": {
        "label": "Pothole",
        "color": "#FB7185",
        "bg": "rgba(251, 113, 133, 0.14)",
        "border": "rgba(251, 113, 133, 0.45)",
    },
}

DEFAULT_STYLE = {
    "color": "#60A5FA",
    "bg": "rgba(96, 165, 250, 0.14)",
    "border": "rgba(96, 165, 250, 0.45)",
}


def inject_custom_css():
  """Inject custom styling for the Streamlit dashboard."""
  st.markdown(
      """
        <style>
            .main .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2.5rem;
            }
            .hero-banner {
                background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #0284C7 100%);
                border: 1px solid rgba(147, 197, 253, 0.28);
                border-radius: 16px;
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.4rem;
                box-shadow: 0 10px 25px rgba(2, 6, 23, 0.35);
            }
            .hero-badge {
                display: inline-block;
                background: rgba(255, 255, 255, 0.14);
                color: #E0F2FE;
                border: 1px solid rgba(224, 242, 254, 0.28);
                padding: 0.22rem 0.7rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                margin-bottom: 0.6rem;
                letter-spacing: 0.3px;
            }
            .hero-title {
                color: #FFFFFF;
                font-size: 1.85rem;
                font-weight: 800;
                margin: 0 0 0.35rem 0;
            }
            .hero-sub {
                color: #DBEAFE;
                font-size: 0.98rem;
                margin: 0;
                line-height: 1.55;
            }
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
                gap: 0.85rem;
                margin: 0.75rem 0 1.2rem 0;
            }
            .kpi-card {
                background: linear-gradient(180deg, rgba(30, 58, 138, 0.22) 0%, rgba(15, 23, 42, 0.65) 100%);
                border: 1px solid rgba(96, 165, 250, 0.3);
                border-radius: 14px;
                padding: 1rem 1.1rem;
            }
            .kpi-label {
                font-size: 0.8rem;
                color: #93C5FD;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 0.3rem;
            }
            .kpi-value {
                font-size: 1.55rem;
                font-weight: 800;
                color: #F8FAFC;
            }
            .class-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
                gap: 0.85rem;
                margin-top: 0.6rem;
            }
            .class-card {
                border-radius: 14px;
                padding: 0.95rem 1.05rem;
                border-width: 1px;
                border-style: solid;
            }
            .class-name {
                font-size: 0.95rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }
            .class-count {
                font-size: 1.45rem;
                font-weight: 800;
                color: #FFFFFF;
                margin-bottom: 0.2rem;
            }
            .class-conf {
                font-size: 0.82rem;
                color: #CBD5E1;
            }
            .section-heading {
                font-size: 1.15rem;
                font-weight: 700;
                color: #E2E8F0;
                border-left: 4px solid #38BDF8;
                padding-left: 0.65rem;
                margin: 1rem 0 0.65rem 0;
            }
        </style>
        """,
      unsafe_allow_html=True,
  )


def ensure_weights_exist(weights_path: Path, weights_url: str) -> bool:
  """Ensure model weights exist locally or download them from the release URL."""
  if weights_path.exists() and weights_path.stat().st_size > 0:
    return True

  if not weights_url:
    return False

  try:
    with st.spinner(
        "Downloading YOLO26m model weights (best.pt) from GitHub Releases..."
    ):
      urllib.request.urlretrieve(weights_url, weights_path)
    return weights_path.exists() and weights_path.stat().st_size > 0
  except Exception as exc:
    st.error(f"Failed to download model weights automatically: {exc}")
    if weights_path.exists():
      weights_path.unlink(missing_ok=True)
    return False


@st.cache_resource(show_spinner=False)
def load_model(weights_path_str: str):
  """Load and cache the YOLO model."""
  return YOLO(weights_path_str)


def summarize_detections(result) -> tuple[int, list[dict]]:
  """Summarize total detections and per-class counts and peak confidence."""
  boxes = result.boxes
  names = result.names or {}

  if boxes is None or len(boxes) == 0:
    return 0, []

  cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()
  confs = boxes.conf.cpu().numpy().tolist()

  counts = Counter(cls_ids)
  max_conf_per_cls = {}
  for cid, c in zip(cls_ids, confs):
    if cid not in max_conf_per_cls or c > max_conf_per_cls[cid]:
      max_conf_per_cls[cid] = c

  summary_rows = []
  for cid, count in counts.most_common():
    raw_label = str(names.get(cid, cid))
    key_lower = raw_label.lower().strip()
    meta = CLASS_META.get(key_lower, {})
    display_label = meta.get("label", raw_label)
    color = meta.get("color", DEFAULT_STYLE["color"])
    bg = meta.get("bg", DEFAULT_STYLE["bg"])
    border = meta.get("border", DEFAULT_STYLE["border"])

    summary_rows.append({
        "raw_key": key_lower,
        "Class": display_label,
        "Detections": count,
        "Max Confidence": f"{max_conf_per_cls[cid]:.2f}",
        "color": color,
        "bg": bg,
        "border": border,
    })

  return len(cls_ids), summary_rows


inject_custom_css()

with st.sidebar:
  st.markdown("### ⚙️ Inference Settings")
  conf_threshold = st.slider(
      "Confidence Threshold (`conf`)",
      min_value=0.05,
      max_value=0.95,
      value=0.25,
      step=0.05,
      help="Minimum confidence score required to keep a detected bounding box.",
  )
  iou_threshold = st.slider(
      "IoU Threshold (`iou`)",
      min_value=0.10,
      max_value=0.90,
      value=0.45,
      step=0.05,
      help="Intersection-over-Union threshold for Non-Maximum Suppression.",
  )
  img_size = st.selectbox(
      "Image Size (`imgsz`)",
      options=[640, 800, 1024],
      index=0,
      help="Inference resolution passed to YOLO26m.",
  )
  use_tta = st.checkbox(
      "Enable Test-Time Augmentation (`TTA`)",
      value=False,
      help="Applies augmentation during inference for potentially stronger recall.",
  )

  st.markdown("---")
  st.markdown("### 🎯 Target Damage Classes")
  st.markdown(
      """
        - **Crack** — Longitudinal / transverse cracks
        - **Alligator** — Interconnected fatigue cracking
        - **Block** — Rectangular interlocking cracks
        - **Pothole** — Pavement surface cavities
        """
  )
  st.caption("Model: **YOLO26m** | Instant AI Diploma — DataSpace #2")

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">COMPUTER VISION • OBJECT DETECTION • YOLO26m</div>
        <h1 class="hero-title">🛣️ Road Damage Detection Inspector</h1>
        <p class="hero-sub">
            Upload a road surface image to detect <strong>crack</strong>, <strong>alligator</strong>,
            <strong>block</strong>, and <strong>pothole</strong> damages, inspect per-class counts, and
            tune inference thresholds in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

weights_ready = ensure_weights_exist(WEIGHTS_PATH, WEIGHTS_URL)
if not weights_ready:
  st.error(
      "Model weights (`best.pt`) were not found locally and could not be"
      " downloaded automatically."
  )
  st.stop()

with st.spinner("Loading YOLO26m model..."):
  model = load_model(str(WEIGHTS_PATH))

uploaded_file = st.file_uploader(
    "📤 Upload a road image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  image_np = np.array(image)

  with st.spinner("Analyzing road surface..."):
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        augment=use_tta,
        verbose=False,
    )

  result = results[0]
  annotated_bgr = result.plot()
  annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
  annotated_pil = Image.fromarray(annotated_rgb)

  total_detections, summary_rows = summarize_detections(result)
  unique_classes = len(summary_rows)
  top_conf = summary_rows[0]["Max Confidence"] if summary_rows else "—"

  st.markdown(
      f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Detections</div>
                <div class="kpi-value">{total_detections}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Damage Types Found</div>
                <div class="kpi-value">{unique_classes}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Peak Confidence</div>
                <div class="kpi-value">{top_conf}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Active Config</div>
                <div class="kpi-value" style="font-size: 1.15rem; padding-top: 0.35rem;">
                    conf={conf_threshold:.2f} | sz={img_size}
                </div>
            </div>
        </div>
        """,
      unsafe_allow_html=True,
  )

  col1, col2 = st.columns(2, gap="large")
  with col1:
    st.markdown(
        '<div class="section-heading">📷 Original Image</div>',
        unsafe_allow_html=True,
    )
    st.image(image, use_container_width=True)

  with col2:
    st.markdown(
        '<div class="section-heading">🎯 Detected Road Damages</div>',
        unsafe_allow_html=True,
    )
    st.image(annotated_pil, use_container_width=True)

  st.markdown(
      '<div class="section-heading">📊 Class Summary</div>',
      unsafe_allow_html=True,
  )

  if summary_rows:
    cards_html = ['<div class="class-grid">']
    for row in summary_rows:
      cards_html.append(
          f"""
                <div class="class-card" style="background: {row['bg']}; border-color: {row['border']};">
                    <div class="class-name" style="color: {row['color']};">{row['Class']}</div>
                    <div class="class-count">{row['Detections']} detected</div>
                    <div class="class-conf">Max confidence: {row['Max Confidence']}</div>
                </div>
                """
      )
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

    with st.expander("📋 View Tabular Summary"):
      st.dataframe(
          [
              {
                  "Class": r["Class"],
                  "Detections": r["Detections"],
                  "Max Confidence": r["Max Confidence"],
              }
              for r in summary_rows
          ],
          use_container_width=True,
          hide_index=True,
      )
  else:
    st.info(
        "No road damages were detected at the current confidence threshold. Try"
        " lowering `conf`, increasing `imgsz`, or enabling `TTA`."
    )

  buf = io.BytesIO()
  annotated_pil.save(buf, format="PNG")
  st.download_button(
      label="⬇️ Download Annotated Image (PNG)",
      data=buf.getvalue(),
      file_name=f"predicted_{Path(uploaded_file.name).stem}.png",
      mime="image/png",
  )
else:
  st.info(
      "👆 Upload a road image above to run **YOLO26m** inference and view"
      " detected damage categories."
  )
