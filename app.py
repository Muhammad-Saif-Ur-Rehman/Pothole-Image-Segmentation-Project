import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP runtime conflict

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------------- CONFIG ----------------
sns.set(style="whitegrid")
st.set_page_config(page_title="Pothole Segmentation", layout="wide")
os.makedirs("outputs", exist_ok=True)

@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model()

def classify_severity(area):
    if area < 1000:
        return 'Low'
    elif area < 5000:
        return 'Medium'
    return 'High'

def analyze_image(image):
    results = model.predict(image, save=False)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []
    boxes = results.boxes
    confidences = boxes.conf.cpu().numpy() if boxes and boxes.conf is not None else []
    areas = [np.sum(mask) for mask in masks]
    severities = [classify_severity(a) for a in areas]

    if not areas:
        return pd.DataFrame(), results.plot()

    df = pd.DataFrame({
        "Pothole #": list(range(1, len(areas) + 1)),
        "Area": areas,
        "Confidence": confidences,
        "Severity": severities
    })

    annotated = results.plot()
    return df, annotated

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Failed to open video file")
        return pd.DataFrame(), ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamp = int(time.time())
    output_path = os.path.join("outputs", f"segmented_output_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_stats = []
    frame_idx = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    show_progress = total_frames > 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, save=False)[0]
        count = len(results.boxes) if results.boxes else 0
        frame_stats.append((frame_idx, count))
        frame_idx += 1

        annotated = results.plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
        
        if show_progress:
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"📹 Processing frame {frame_idx}/{total_frames} ({progress*100:.1f}%)")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    if not os.path.exists(output_path):
        st.error("❌ Segmented video not found.")
        return pd.DataFrame(), ""

    df = pd.DataFrame(frame_stats, columns=["Frame", "Pothole Count"])
    return df, output_path

def show_charts(df):
    if df.empty:
        st.warning("⚠️ No potholes detected.")
        return

    st.subheader("📊 Pothole Area (Bar Chart)")
    fig1, ax1 = plt.subplots()
    sns.barplot(data=df, x="Pothole #", y="Area", hue="Severity", palette="Blues_d", ax=ax1)
    st.pyplot(fig1)

    st.subheader("🧯 Severity Distribution (Pie Chart)")
    fig2, ax2 = plt.subplots()
    severity_counts = df['Severity'].value_counts()
    ax2.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set2"), startangle=90)
    ax2.axis("equal")
    st.pyplot(fig2)

    st.subheader("📈 Confidence Scores (Histogram)")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["Confidence"], bins=10, kde=True, color="coral", ax=ax3)
    st.pyplot(fig3)

def show_line_chart(df):
    if df.empty:
        st.warning("⚠️ No pothole data found from video.")
        return
    st.subheader("📽️ Pothole Count per Frame")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Frame", y="Pothole Count", marker="o", ax=ax)
    st.pyplot(fig)

# ---------------- UI ----------------
st.title("🕳️ Pothole Detection using YOLOv8 (Image + Video Segmentation)")
mode = st.sidebar.radio("Choose Inference Type", ["Image", "Video"])

if mode == "Image":
    uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Running segmentation..."):
            df, annotated = analyze_image(image)

        st.image(annotated, caption="🧠 Segmented Output", use_column_width=True)
        st.markdown(f"**✅ Total Potholes Detected:** `{len(df)}`")
        if not df.empty:
            st.markdown(f"**📊 Avg. Confidence Score:** `{df['Confidence'].mean():.2f}`")
        show_charts(df)

elif mode == "Video":
    uploaded_video = st.file_uploader("📤 Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        st.video(temp_path)

        with st.spinner("🔍 Analyzing and segmenting video..."):
            video_df, segmented_path = analyze_video(temp_path)

        try:
            os.unlink(temp_path)
        except Exception as e:
            st.warning(f"⚠️ Could not delete temp file: {e}")

        if not video_df.empty:
            show_line_chart(video_df)

        # Removed: st.video(segmented_path) — as requested

        st.subheader("💾 Download Segmented Video")
        if segmented_path and os.path.exists(segmented_path):
            with open(segmented_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Segmented Video",
                    data=f,
                    file_name="segmented_potholes.mp4",
                    mime="video/mp4"
                )
        else:
            st.error("❌ Segmented video not found or failed to save.")
