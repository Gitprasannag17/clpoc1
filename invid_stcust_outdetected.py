#0. Set up authentication for local development
import os
from google.cloud import storage
import cv2
from google.cloud import vision
import imagehash
from PIL import Image
import io


if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.system("gcloud auth application-default login")

# Initialize Google Cloud Clients
vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()

#1.Load Customer Images from Cloud Storage
#Retrieve all customer images from the customer base bucket:
#from google.cloud import storage

#storage_client = storage.Client()
#CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"

def list_customer_images():
    
    print(f"started list_customer_images")
    CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"    
    """Lists all images in the customer base bucket"""
    bucket = storage_client.bucket(CUSTOMER_BUCKET)
    print(f"finished list_customer_images")
    return [blob.name for blob in bucket.list_blobs() if blob.name.endswith(('.jpg', '.png'))]

#customer_images = list_customer_images()
#print("Customer Images:", customer_images)

#2. Extract Faces from Video using Google Cloud Vision
#Use OpenCV to extract frames from the video, then use Google Cloud Vision to detect faces.
#import cv2
#from google.cloud import vision

#vision_client = vision.ImageAnnotatorClient()

def extract_faces_from_video(video_path):
    print(f"started extract_faces_from_video")
    """Extracts faces from video frames"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  # Process every 30th frame
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            # Send image to Google Cloud Vision API
            image = vision.Image(content=img_bytes)
            response = vision_client.face_detection(image=image)

            if response.face_annotations:
                detected_faces.append(frame)

    cap.release()
    print(f"finished extract_faces_from_video")    
    return detected_faces

#3. Match Detected Faces with Customer Base Images
#Compare extracted faces with customer images stored in the bucket using image hashing.

#import imagehash
#from PIL import Image
#import io

def get_image_hash(image_content):
    print(f"started get_image_hash")    
    """Returns hash of an image for comparison"""
    image = Image.open(io.BytesIO(image_content))
    print(f"finished get_image_hash")        
    return imagehash.average_hash(image)

def download_customer_images():
    print(f"started download_customer_images")     

    customer_images = list_customer_images()   
    #"""Download customer images from the bucket for matching"""
    customer_hashes = {}

    CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"    
    bucket = storage_client.bucket(CUSTOMER_BUCKET)

    for image_name in customer_images:
        blob = bucket.blob(image_name)
        image_bytes = blob.download_as_bytes()
        customer_hashes[image_name] = get_image_hash(image_bytes)
        print(f"hash: {customer_hashes[image_name]}")
    print(f"finished download_customer_images")    
    return customer_hashes

#customer_hashes = download_customer_images()

def match_faces(detectedfaces):
    print(f"started match_faces")  

    customer_hashes = download_customer_images()      
    """Match detected faces with stored customer images"""
    matched_customers = set()
  
    for face in detectedfaces:
        face_hash = get_image_hash(cv2.imencode('.jpg', face)[1].tobytes())

        for customer_name, customer_hash in customer_hashes.items():
            if face_hash - customer_hash < 5:  # 5 is the similarity threshold
                matched_customers.add(customer_name)
    print(f"finished match_faces") 
    print(f"{len(matched_customers)}")     
    return matched_customers

#matched_customers = match_faces(detected_faces)
#print("Matched Customers:", matched_customers)

#4. Upload Matched Customer Images to Output Bucket
#If a customer appears in the video, upload their corresponding image from the customer base bucket.

#OUTPUT_BUCKET = "clpoc1-customerdetected-bucket"

def upload_matched_customers(faces):
    print(f"started upload_matched_customers")     

    matched_customers = match_faces(faces)
    print(f"Detected {len(matched_customers)} matched_customers") 

    CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"    
    
    """Uploads matched customer images to another bucket"""
    OUTPUT_BUCKET = "clpoc1-customerdetected-bucket"
    output_bucket = storage_client.bucket(OUTPUT_BUCKET)
    print(f"OUTPUT BUCKET is {OUTPUT_BUCKET}")    

    for customer in matched_customers:
        blob = storage_client.bucket(CUSTOMER_BUCKET).blob(customer)
        new_blob = output_bucket.blob(customer)
        new_blob.upload_from_string(blob.download_as_bytes())
        print(f"Uploaded {customer} to {OUTPUT_BUCKET}")

    print(f"finished upload_matched_customers") 

#upload_matched_customers()
# Main function to process video and identify customers
def process_video(video_filename):
    try:
        print(f"Processing video: {video_filename}")

        # Download video from Cloud Storage
        VIDEO_BUCKET="clpoc1-input-video-bucket"

        bucket = storage_client.bucket(VIDEO_BUCKET)
        blob = bucket.blob(video_filename)
        local_video_path = f"/{video_filename}"
        blob.download_to_filename(local_video_path) #

        # Extract faces
        faces = extract_faces_from_video(local_video_path)
        print(f"Detected {len(faces)} faces.")

        #customer_images = list_customer_images()

        #customer_hashes = download_customer_images()

        #matched_customers = match_faces(faces)

        #CUSTOMER_BUCKET = "clpoc1-customerbase-bucket"

        upload_matched_customers(faces)

    except Exception as e:
        print(f"Error processing video: {e}")
                                           
# Run the function
process_video("nitaakashshloka.mp4")