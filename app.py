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
  """يتحقق من وجود ملف الأوزان محلياً أو يحمّله تلقائياً من GitHub Releases."""
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
  """تحميل نموذج YOLO وتخزينه في الذاكرة المؤقتة."""
  return YOLO(weights_path_str)


def summarize_detections(result) -> tuple[int, list[dict]]:
  """استخراج إجمالي الص
