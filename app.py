import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image
)

from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Exam Hall Monitoring AI",
    page_icon="🎓",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "Exam Hall Monitoring.v3i.yolov8"
    / "best.pt"
)


# ==========================================
# DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #070b14;
    color: #e8eefc;
}

[data-testid="stSidebar"] {
    background: #10192a;
}

[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #304568;
    padding: 15px;
    border-radius: 12px;
}

h1, h2, h3 {
    color: #e4edff;
}

.hero {
    padding: 25px;
    background: linear-gradient(120deg, #182d4c, #11182b);
    border: 1px solid #304568;
    border-radius: 18px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


if not MODEL_PATH.is_file():
    st.error("best.pt model was not found.")
    st.code(str(MODEL_PATH))
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🎓 Exam Hall Monitoring AI</h1>
    <p>YOLO Detection | Video Analytics | Live Monitoring</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("⚙️ Settings")
    st.success("YOLO Model Loaded")

    confidence = st.slider(
        "Confidence Threshold",
        0.10, 0.90, 0.50, 0.05
    )

    image_size = st.select_slider(
        "Image Size",
        options=[320, 416, 512, 640, 768],
        value=640
    )

    device = st.selectbox(
        "Processing Device",
        ["mps", "cpu"]
    )

    st.caption("Use MPS on your Apple Silicon Mac.")


# ==========================================
# LIVE CAMERA PROCESSOR
# ==========================================

class CameraProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = model.predict(
            image,
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ==========================================
# TWO TABS
# ==========================================

video_tab, live_tab = st.tabs([
    "🎞️ Video Upload",
    "📡 Live Stream"
])


# ==========================================
# TAB 1: VIDEO UPLOAD
# ==========================================

with video_tab:

    st.subheader("🎞️ Upload Exam Hall Video")

    video = st.file_uploader(
        "Choose your video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    if video is not None:

        video_bytes = video.getvalue()
        st.video(video_bytes)

        if st.button("▶ Start Video Analysis", type="primary"):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                # Save input video
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(video.name).suffix or ".mp4"
                ) as temp:
                    temp.write(video_bytes)
                    input_path = temp.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    raise RuntimeError("Could not open uploaded video.")

                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                st.write(f"Total Frames: {total_frames}")
                st.write(f"FPS: {fps:.2f}")

                # Create output video
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as output:
                    output_path = output.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError("Could not create output video.")

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                class_counts = Counter()
                detection_records = []

                total_detections = 0
                processed_frames = 0

                start_time = time.time()

                # Process video frames
                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    results = model.predict(
                        frame,
                        conf=confidence,
                        imgsz=image_size,
                        device=device,
                        verbose=False
                    )

                    result = results[0]

                    for box in result.boxes:

                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = float(box.conf[0])

                        class_counts[class_name] += 1
                        total_detections += 1

                        detection_records.append({
                            "Frame": processed_frames + 1,
                            "Class": class_name,
                            "Confidence": round(conf, 4)
                        })

                    annotated = result.plot()

                    writer.write(annotated)

                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        preview.image(
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="YOLO Detection Preview"
                        )

                    if total_frames > 0:
                        progress.progress(
                            min(processed_frames / total_frames, 1.0)
                        )

                    status.write(
                        f"Processing frame {processed_frames}/{total_frames}"
                    )

                elapsed = time.time() - start_time

                cap.release()
                cap = None

                writer.release()
                writer = None

                progress.progress(1.0)

                st.success("Video analysis completed!")

                # ==================================
                # ANALYTICS REPORT
                # ==================================

                st.header("📊 Analytics Report")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    f"{processed_frames:,}"
                )

                col2.metric(
                    "Total Detections",
                    f"{total_detections:,}"
                )

                col3.metric(
                    "Processing Time",
                    f"{elapsed:.1f} sec"
                )

                st.subheader("Detection Counts")

                if class_counts:

                    for name, count in class_counts.items():
                        st.write(f"**{name}:** {count}")

                else:
                    st.info("No objects detected.")

                # Detailed analytics
                if detection_records:

                    df = pd.DataFrame(detection_records)

                    st.divider()
                    st.subheader("🎯 Detection Statistics")

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Average Confidence",
                        f"{df['Confidence'].mean() * 100:.2f}%"
                    )

                    c2.metric(
                        "Detected Classes",
                        df["Class"].nunique()
                    )

                    st.subheader("📈 Detection Distribution")

                    chart_data = (
                        df["Class"]
                        .value_counts()
                        .rename_axis("Class")
                        .to_frame("Detections")
                    )

                    st.bar_chart(chart_data)

                    st.subheader("📋 Detection Records")

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                    # Download CSV
                    csv_data = df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download Analytics CSV",
                        data=csv_data,
                        file_name="exam_analytics_report.csv",
                        mime="text/csv"
                    )

                    # PDF_REPORT_GENERATOR
                    st.divider()
                    st.subheader("📄 Professional PDF Report")

                    if st.button("Generate PDF Report", type="primary"):
                        pdf_buffer = BytesIO()

                        styles = getSampleStyleSheet()
                        story = []

                        doc = SimpleDocTemplate(
                            pdf_buffer,
                            pagesize=letter,
                            rightMargin=36,
                            leftMargin=36,
                            topMargin=36,
                            bottomMargin=36
                        )

                        story.append(Paragraph(
                            "Exam Hall Monitoring AI",
                            styles["Title"]
                        ))

                        story.append(Paragraph(
                            "Video Detection and Analytics Report",
                            styles["Heading2"]
                        ))

                        story.append(Spacer(1, 15))

                        report_date = datetime.now().strftime(
                            "%d %B %Y, %I:%M %p"
                        )

                        summary = [
                            ["Report Date", report_date],
                            ["Frames Processed", str(processed_frames)],
                            ["Total Detections", str(total_detections)],
                            ["Processing Time", f"{elapsed:.2f} seconds"],
                            [
                                "Average Confidence",
                                f"{df['Confidence'].mean() * 100:.2f}%"
                            ],
                            ["Detected Classes", str(df["Class"].nunique())]
                        ]

                        story.append(Paragraph(
                            "Analysis Summary",
                            styles["Heading2"]
                        ))

                        summary_table = Table(
                            [["Metric", "Result"]] + summary,
                            colWidths=[220, 250]
                        )

                        summary_table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B5A")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("PADDING", (0, 0), (-1, -1), 8),
                        ]))

                        story.append(summary_table)
                        story.append(Spacer(1, 20))

                        story.append(Paragraph(
                            "Detection Counts by Class",
                            styles["Heading2"]
                        ))

                        count_data = [["Class", "Detections"]]

                        for name, count in class_counts.items():
                            count_data.append([str(name), str(count)])

                        count_table = Table(
                            count_data,
                            colWidths=[220, 250]
                        )

                        count_table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B5A")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("PADDING", (0, 0), (-1, -1), 8),
                        ]))

                        story.append(count_table)
                        story.append(Spacer(1, 20))

                        # Create detection chart
                        fig, ax = plt.subplots(figsize=(7, 3.5))

                        ax.bar(
                            list(class_counts.keys()),
                            list(class_counts.values())
                        )

                        ax.set_title("Detection Distribution")
                        ax.set_xlabel("Class")
                        ax.set_ylabel("Detection Count")
                        ax.tick_params(axis="x", rotation=20)

                        fig.tight_layout()

                        chart_buffer = BytesIO()
                        fig.savefig(
                            chart_buffer,
                            format="png",
                            dpi=150,
                            bbox_inches="tight"
                        )
                        plt.close(fig)
                        chart_buffer.seek(0)

                        story.append(Paragraph(
                            "Detection Distribution Chart",
                            styles["Heading2"]
                        ))

                        story.append(Image(
                            chart_buffer,
                            width=6.5 * inch,
                            height=3.2 * inch
                        ))

                        story.append(Spacer(1, 15))

                        # Add sample detection records
                        story.append(Paragraph(
                            "Detection Records (First 30)",
                            styles["Heading2"]
                        ))

                        record_data = [
                            ["Frame", "Class", "Confidence"]
                        ]

                        for _, row in df.head(30).iterrows():
                            record_data.append([
                                str(row["Frame"]),
                                str(row["Class"]),
                                f"{row['Confidence'] * 100:.2f}%"
                            ])

                        record_table = Table(
                            record_data,
                            repeatRows=1,
                            colWidths=[100, 180, 150]
                        )

                        record_table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B5A")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("PADDING", (0, 0), (-1, -1), 6),
                        ]))

                        story.append(record_table)

                        story.append(Spacer(1, 15))

                        story.append(Paragraph(
                            "Note: These are automated YOLO predictions, "
                            "not verified incidents of cheating.",
                            styles["Italic"]
                        ))

                        doc.build(story)

                        pdf_buffer.seek(0)

                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name="Exam_Hall_Analytics_Report.pdf",
                            mime="application/pdf",
                            key="download_pdf_report"
                        )


                # Download annotated video
                with open(output_path, "rb") as file:
                    output_bytes = file.read()

                st.download_button(
                    "⬇️ Download Annotated Video",
                    data=output_bytes,
                    file_name="exam_analysis.mp4",
                    mime="video/mp4"
                )

            except Exception as e:

                st.error(f"Video processing error: {e}")

            finally:

                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

                if output_path and os.path.exists(output_path):
                    os.remove(output_path)


# ==========================================
# TAB 2: LIVE STREAM
# ==========================================

with live_tab:

    st.subheader("📡 Live Camera Monitoring")

    st.info(
        "Allow camera permission in your browser to start detection."
    )

    webrtc_streamer(
        key="exam-hall-live",
        video_processor_factory=CameraProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.caption(
        "Live camera detection uses your existing YOLO model."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Exam Hall Monitoring AI | Powered by YOLO & Streamlit"
)
