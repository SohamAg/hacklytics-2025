import streamlit as st
import cv2
import os
import json
import numpy as np
from collections import deque
from ultralytics import YOLO
from openai import OpenAI
import openai
import subprocess

# ------------------- SETUP ------------------- #
global inter
# YOLO Model
YOLO_MODEL_PATH = "datasets/final-training/runs/detect/train/weights/best.pt"
model = YOLO(YOLO_MODEL_PATH)

# OpenAI API (Local LLM)
openai.api_base = "http://127.0.0.1:1234"
openai.api_key = ""
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="unneeded")

# Directories
FRAME_FOLDER = "extracted_frames"
OUTPUT_FOLDER = "yolo_output_frames"
PROCESSED_VIDEO = "output_vid.mp4"

# Ensure folders exist
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------- FRAME EXTRACTION ------------------- #

def extract_frames(video_path):
    """ Extracts frames from video and saves them to 'extracted_frames' folder. """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(FRAME_FOLDER, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    return frame_count

# ------------------- YOLO DETECTION ON FRAMES ------------------- #

def clear_old_data():
    """ Removes old frames, processed images, and output video before new processing. """
    # Remove old frames
    for folder in [FRAME_FOLDER, OUTPUT_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Remove old processed video
    if os.path.exists(PROCESSED_VIDEO):
        os.remove(PROCESSED_VIDEO)

    print("🗑️ Cleared old frames and video!")


def run_yolo_on_frames():
    global inter
    """ Runs YOLO detection on extracted frames and saves processed frames. Also returns total interferences. """
    frame_files = sorted([f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg")])

    if not frame_files:
        print("❌ No frames found for YOLO processing!")
        return 0, 0  # Return 0 processed frames, 0 total interferences

    processed_count = 0
    total_interferences = 0  # Track total interference count

    for frame_file in frame_files:
        frame_path = os.path.join(FRAME_FOLDER, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"⚠️ Could not read frame: {frame_file}")
            continue

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                conf = box.conf[0].item()  
                cls = int(box.cls[0].item())  
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                total_interferences += 1  # Increment total interferences count

        output_path = os.path.join(OUTPUT_FOLDER, frame_file)
        cv2.imwrite(output_path, frame)
        processed_count += 1

    print(f"✅ Processed {processed_count} frames with YOLO!")
    # print(f"⚡ Total Interference Events Detected: {total_interferences}")
    inter = total_interferences

    return processed_count  # Return both values


# ------------------- REGENERATE VIDEO ------------------- #
def convert_video_to_h264(input_video, output_video):
    """ Converts OpenCV-generated MP4 to H.264 codec for Streamlit compatibility """
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_video, "-vcodec", "libx264", "-crf", "23", output_video
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def generate_output_video():
    """ Compiles processed frames into a final video and converts it to H.264 """
    frame_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".jpg")])

    if not frame_files:
        print("❌ No processed frames found! Cannot generate video.")
        return None

    first_frame = cv2.imread(os.path.join(OUTPUT_FOLDER, frame_files[0]))

    if first_frame is None:
        print("❌ Could not read the first frame!")
        return None

    height, width, _ = first_frame.shape
    fps = 30  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # OpenCV's default MP4 codec
    out = cv2.VideoWriter(PROCESSED_VIDEO, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(OUTPUT_FOLDER, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {frame_file}")
            continue

        out.write(frame)

    out.release()
    print("✅ Processed video saved successfully!")

    # Convert to H.264 codec for Streamlit compatibility
    converted_video = "final_output_h264.mp4"
    convert_video_to_h264(PROCESSED_VIDEO, converted_video)

    return converted_video

# ------------------- HEURISTIC DPI/OPI DETECTION ------------------- #

def apply_heuristics():
    """ Apply NFL rules to classify DPI/OPI based on frame timing, proximity, and movement. """
    with open("yolo_detected_frames.json") as f:
        interference_frames = json.load(f)

    total_interference = len(interference_frames.keys())  # Count total interference events

    # 🚨 If <10 interferences, inform the LLM directly
    if total_interference < 10:
        heuristic_data = {
            "dpi_frames": 0,
            "opi_frames": 0,
            "sustained_contact_frames": 0,
            "head_turn_before_contact": False,
            "proximity_at_contact": 0,
            "last_contact_frame": -1,
            "ball_arrival_frame": 50,
            "low_interference": True  # 🚀 Special flag for LLM
        }
        return [], [], 0.0, heuristic_data

    DPI_frames, OPI_frames = [], []
    BALL_ARRIVAL_FRAME = 50  
    confidence_factor = 0  
    sustained_contact = 0  
    total_sustained_contact = 0  
    head_turn_before_contact = False  
    proximity_at_contact = 0  
    last_contact_frame = -1  

    MIN_CONTACT_DURATION = 5  
    PROXIMITY_THRESHOLD = 50  

    for frame_num in interference_frames.keys():
        frame_num = int(frame_num)

        # Track last interference frame
        last_contact_frame = frame_num  

        if frame_num < BALL_ARRIVAL_FRAME:
            DPI_frames.append(frame_num)
            confidence_factor += 0.2  

        else:
            OPI_frames.append(frame_num)
            confidence_factor += 0.15  

        # Track sustained contact
        if frame_num - 1 in interference_frames:
            sustained_contact += 1
        else:
            sustained_contact = 0  

        if sustained_contact >= MIN_CONTACT_DURATION:
            confidence_factor += 0.1  
            total_sustained_contact += sustained_contact  

        # Check if the interference happened close to the ball arrival frame
        if BALL_ARRIVAL_FRAME - 10 < frame_num < BALL_ARRIVAL_FRAME + 10:
            confidence_factor += 0.05  

    confidence_factor = min(confidence_factor, 1.0)

    heuristic_data = {
        "dpi_frames": len(DPI_frames),
        "opi_frames": len(OPI_frames),
        "sustained_contact_frames": total_sustained_contact,
        "head_turn_before_contact": head_turn_before_contact,
        "proximity_at_contact": proximity_at_contact,
        "last_contact_frame": last_contact_frame,
        "ball_arrival_frame": BALL_ARRIVAL_FRAME,
        "low_interference": False  # Normal case
    }

    return DPI_frames, OPI_frames, confidence_factor, heuristic_data



# ------------------- LLM DPI/OPI QUERY ------------------- #

def query_llm(heuristic_data):
    global inter
    """ Uses an LLM to classify interference using heuristic variables. """
    sys_msg = open("piprompt.txt").read()

    if heuristic_data.get("low_interference", False) or inter < 5:
        user_input = """
        The play has **less than 10 interference events detected** by YOLO.
        Based on NFL rules, does this indicate a **No Call**?
        Explain why minimal interference does not warrant DPI or OPI.
        """
        inter = 0
    else:
        user_input = f"""
        Analyze the following play for pass interference (PI) classification:

        - **DPI Frames:** {heuristic_data['dpi_frames']}
        - **OPI Frames:** {heuristic_data['opi_frames']}
        - **Sustained Contact Frames:** {heuristic_data['sustained_contact_frames']}
        - **Defender Head Turn Before Contact:** {heuristic_data['head_turn_before_contact']}
        - **Proximity at Contact (pixels):** {heuristic_data['proximity_at_contact']}
        - **Last Contact Frame:** {heuristic_data['last_contact_frame']}
        - **Ball Arrival Frame:** {heuristic_data['ball_arrival_frame']}

        Based on NFL rules, determine:
        - Is this Defensive Pass Interference (DPI), Offensive Pass Interference (OPI), or No Call?
        - Explain why based on the given data.
        """

    try:
        completion = client.chat.completions.create(
            model="internlm/internlm2_5-20b-chat-gguf/",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_input}
            ],
        )

        llm_response = completion.choices[0].message.content.strip()
        return llm_response

    except Exception as e:
        print(f"❌ LLM Query Failed: {e}")
        return "⚠️ Error querying LLM."


# ------------------- STREAMLIT UI ------------------- #

st.title("🏈 Faircall AI - Pass Interference Detection")

uploaded_file = st.file_uploader("Upload an NFL Play Video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

if st.button("🔍 Process Video"):
    clear_old_data()
    with st.spinner("Extracting frames..."):
        total_frames = extract_frames(video_path)
        st.write(f"✅ Extracted {total_frames} frames.")

    with st.spinner("Running YOLO detection on frames..."):
        processed_frames = run_yolo_on_frames()
        st.write(f"✅ YOLO processed {processed_frames} frames.")

    with st.spinner("Applying Heuristic Rules..."):
        dpi_frames, opi_frames, confidence_factor, data = apply_heuristics()

    with st.spinner("Querying LLM for DPI/OPI Decision..."):
        llm_decision = query_llm(data)
        st.subheader("🤖 LLM Decision:")
        st.write(llm_decision)

    with st.spinner("Compiling processed video..."):
        output_video = generate_output_video()

    st.success("🚀 Processing Complete!")

    if output_video:
        st.subheader("📹 Processed Video Output")
        st.video(output_video)
