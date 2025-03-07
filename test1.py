import cv2
import os
from google.cloud import storage, vision
from deepface import DeepFace

# Set up authentication for local development
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.system("gcloud auth application-default login")

# Initialize Google Cloud Clients
storage_client = storage.Client()
vision_client = vision.ImageAnnotatorClient()

# Define Cloud Storage Buckets
VIDEO_BUCKET = "clpoc1-input-video-bucket"
CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"
OUTPUT_BUCKET = "clpoc1-customerdetected-bucket"

def extract_faces_from_video(video_path):
    """Extracts frames from a video and checks for faces."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing Frame {frame_count}")  # Debugging print

        if frame_count % 30 == 0:  # Process every 30th frame
            frame_path = os.path.join(TEMP_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)  # Save frame for debugging
            print(f"Saved Frame {frame_count} to {frame_path}")

    cap.release()
    return extracted_faces
