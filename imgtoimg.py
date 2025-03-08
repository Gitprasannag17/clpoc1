#this is not matching the images correctly :
import os
import cv2
import numpy as np
import io
from google.cloud import vision, storage
#from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean


# Set up authentication for local development
#if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
#    os.system("gcloud auth application-default login")

# Initialize Google Cloud Clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

# Define GCP Storage buckets
INPUTIMAGE_BUCKET = "clpoc1-input-image-bucket"
CUSTOMER_IMAGES_BUCKET = "clpoc1-customerbase-bucket0"
OUTPUT_BUCKET = "clpoc1-customerdetected-bucket0"

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
            embedding = np.array([face.detection_confidence] + 
                                 [lm.position.x for lm in face.landmarks] + 
                                 [lm.position.y for lm in face.landmarks] + 
                                 [lm.position.z for lm in face.landmarks])
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
    best_score = float("inf")  # Euclidean distance: lower is better

    for name, known_embedding in known_faces.items():
        score = euclidean(face_embedding, known_embedding)  # Use Euclidean instead of Cosine
        if score < best_score:
            best_score = score
            best_match = name            

    return best_match, best_score

# Main function to process test images and identify customers
def process_images():
    try:
        print(f"Processing test images: ")

        # Load known customer embeddings
        known_faces = load_customer_faces()

        # Process each inputimage
        testimage_faces = {}
        bucket = storage_client.bucket(INPUTIMAGE_BUCKET)
        blobs = bucket.list_blobs()

        results = []

        for blob in blobs:
            image_bytes = blob.download_as_bytes()
            image = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            #results = []
            testface_embeddings = detect_faces(image)

            for embedding in testface_embeddings:
                match_name, similarity_score = identify_person(embedding, known_faces)

                if similarity_score < 0.3:  # Adjust threshold based on accuracy
                    results.append((image, match_name))

        # Save matched images
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)
        for i, (image, name) in enumerate(results):
            output_path = f"matched_{i}_{name}.jpg"
            _, encoded_image = cv2.imencode(".jpg", image)
            blob = output_bucket.blob(output_path)
            blob.upload_from_string(encoded_image.tobytes(), content_type="image/jpeg")

        print(f"Processed test images and saved matched images.")

    except Exception as e:
        print(f"Error processing test images: {e}")

# Run the function
process_images()
