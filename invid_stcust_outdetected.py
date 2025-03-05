import os
import cv2
import numpy as np
import io
from google.cloud import vision, storage
from scipy.spatial.distance import cosine


# Set up authentication for local development
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.system("gcloud auth application-default login")

# Initialize Google Cloud Clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

# Define GCP Storage buckets
VIDEO_BUCKET = "clpoc1-input-video-bucket"
CUSTOMER_IMAGES_BUCKET = "clpoc1-customerbase-bucket"
OUTPUT_BUCKET = "clpoc1-customerdetected-bucket"

# Function to extract frames from a video
def extract_frames_from_video(video_path, interval=5):
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        if frame_count % interval == 0:
            frames.append(frame)

        frame_count += 1

    video.release()
    return frames

# Function to detect faces in an image using Google Cloud Vision API
def detect_faces(image):
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = encoded_image.tobytes()
    vision_image = vision.Image(content=image_bytes)

    response = vision_client.face_detection(image=vision_image)
    faces = response.face_annotations

    face_embeddings = []
    for face in faces:
        if face.landmarks:
            embedding = np.array([
                face.detection_confidence,
                face.landmarks[0].position.x,
                face.landmarks[0].position.y,
                face.landmarks[0].position.z,
            ])
            face_embeddings.append(embedding)

    return face_embeddings

# Function to load customer images and extract embeddings
def load_customer_faces():
    known_faces = {}
    bucket = storage_client.bucket(CUSTOMER_IMAGES_BUCKET)
    blobs = bucket.list_blobs()

    for blob in blobs:
        image_bytes = blob.download_as_bytes()
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        embeddings = detect_faces(image)
        if embeddings:
            known_faces[blob.name] = embeddings[0]  # Assume 1 face per image

    return known_faces

# Function to match a detected face with known customers
def identify_person(face_embedding, known_faces):
    best_match = None
    best_score = float("inf")

    for name, known_embedding in known_faces.items():
        score = cosine(face_embedding, known_embedding)
        if score < best_score:
            best_score = score
            best_match = name

    return best_match, best_score

# Main function to process video and identify customers
def process_video(video_filename):
    try:
        print(f"Processing video: {video_filename}")

        # Download video from Cloud Storage
        bucket = storage_client.bucket(VIDEO_BUCKET)
        blob = bucket.blob(video_filename)
        local_video_path = f"/{video_filename}"
        blob.download_to_filename(local_video_path)

        # Extract frames
        frames = extract_frames_from_video(local_video_path, interval=10)

        # Load known customer embeddings
        known_faces = load_customer_faces()

        # Process each frame
        results = []
        for idx, frame in enumerate(frames):
            face_embeddings = detect_faces(frame)

            for embedding in face_embeddings:
                match_name, similarity_score = identify_person(embedding, known_faces)

                if similarity_score < 0.3:  # Adjust threshold based on accuracy
                    results.append((frame, match_name))

        # Save matched images
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)
        for i, (image, name) in enumerate(results):
            output_path = f"matched_{i}_{name}.jpg"
            _, encoded_image = cv2.imencode(".jpg", image)
            blob = output_bucket.blob(output_path)
            blob.upload_from_string(encoded_image.tobytes(), content_type="image/jpeg")

        print(f"Processed video '{video_filename}' and saved matched images.")

    except Exception as e:
        print(f"Error processing video: {e}")

# Run the function
process_video("nitaakashshloka.mp4")
