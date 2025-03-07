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
video_filename="nitaakashshloka.mp4"

# Temporary storage for processing images
TEMP_DIR = "/tmp/" if os.name != "nt" else "C:\\temp\\"  # Adjust for Windows

# Ensure temp directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def list_customer_images():
    """Lists all images in the customer base bucket."""
    print(f"Fetching images from: {CUSTOMER_BUCKET}")  # Debugging print
    bucket = storage_client.bucket(CUSTOMER_BUCKET)
    return [blob.name for blob in bucket.list_blobs() if blob.name.endswith(('.jpg', '.png'))]


def detect_faces(image_bytes):
    """Send image to Google Cloud Vision API for face detection"""
    image = vision.Image(content=image_bytes)
    response = vision_client.face_detection(image=image)
    
    if response.face_annotations:
        print(f"Faces detected: {len(response.face_annotations)} ✅")
        return True
    else:
        print("No faces detected in frame ❌")
        return False


def extract_faces_from_video(video_path):
    """Extracts faces from a video and saves them as images."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            frame = cv2.resize(frame, (800, 600))  # Resize to improve detection
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            if detect_faces(img_bytes):  # Call Google Vision API
                frame_path = os.path.join(TEMP_DIR, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_faces.append(frame_path)

    cap.release()
    print(f"Extracted {len(extracted_faces)} frames with faces.")
    return extracted_faces


def download_customer_images():
    """Download customer images from Cloud Storage for matching."""
    customer_images = list_customer_images()
    local_customer_images = {}

    for image_name in customer_images:
        blob = storage_client.bucket(CUSTOMER_BUCKET).blob(image_name)
        local_path = os.path.join(TEMP_DIR, image_name)
        blob.download_to_filename(local_path)
        local_customer_images[image_name] = local_path

    print(f"debug1")
    return local_customer_images


def match_faces(detected_faces, customer_images):
    """Match detected faces with stored customer images using DeepFace embeddings."""
    matched_customers = set()

    for face_path in detected_faces:
        for customer_name, customer_path in customer_images.items():
            try:
                result = DeepFace.verify(face_path, customer_path, model_name="Facenet", enforce_detection=False)
                if result["verified"]:
                    matched_customers.add(customer_name)
                    print(f"✅ Match Found: {customer_name}")
            except Exception as e:
                print(f"Error processing {customer_name}: {e}")

    return matched_customers


def upload_matched_customers(matched_customers):
    """Uploads matched customer images to the output bucket."""
    output_bucket = storage_client.bucket(OUTPUT_BUCKET)

    for customer in matched_customers:
        blob = storage_client.bucket(CUSTOMER_BUCKET).blob(customer)
        new_blob = output_bucket.blob(customer)
        new_blob.upload_from_string(blob.download_as_bytes())
        print(f"🚀 Uploaded {customer} to {OUTPUT_BUCKET}")


# Main Execution
if __name__ == "__main__":
    bucket = storage_client.bucket(VIDEO_BUCKET)
    blob = bucket.blob(video_filename)
    video_path = f"/{video_filename}"

    # Step 1: Extract Faces from Video
    detected_faces = extract_faces_from_video(video_path)

    # Step 2: Download Customer Images
    customer_images = download_customer_images()

    # Step 3: Match Faces with Customer Base
    matched_customers = match_faces(detected_faces, customer_images)

    # Step 4: Upload Matched Customers' Images to Output Bucket
    upload_matched_customers(matched_customers)
