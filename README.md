# clpoc1

Got it! You need to upload images from your **customer base bucket** for only those customers who appear in the **input video**.  

### **Steps to Achieve This:**
1. **Extract frames from the video** and identify faces using **Google Cloud Vision API**.
2. **Compare extracted faces with stored images in the customer base bucket**.
3. **If a match is found, upload the corresponding customer image from the customer base bucket**.

---

### **Implementation Plan**
#### **1. Load Customer Images from Cloud Storage**
Retrieve all customer images from the customer base bucket:
```python
from google.cloud import storage

storage_client = storage.Client()
CUSTOMER_BUCKET = "your-customer-base-bucket"

def list_customer_images():
    """Lists all images in the customer base bucket"""
    bucket = storage_client.bucket(CUSTOMER_BUCKET)
    return [blob.name for blob in bucket.list_blobs() if blob.name.endswith(('.jpg', '.png'))]

customer_images = list_customer_images()
print("Customer Images:", customer_images)
```

---

#### **2. Extract Faces from Video using Google Cloud Vision**
Use OpenCV to extract frames from the video, then use Google Cloud Vision to detect faces.

```python
import cv2
from google.cloud import vision

vision_client = vision.ImageAnnotatorClient()

def extract_faces_from_video(video_path):
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
    return detected_faces

video_path = "local_video.mp4"
faces = extract_faces_from_video(video_path)
print(f"Detected {len(faces)} faces.")
```

---

#### **3. Match Detected Faces with Customer Base Images**
Compare extracted faces with customer images stored in the bucket using **image hashing**.

```python
import imagehash
from PIL import Image
import io

def get_image_hash(image_content):
    """Returns hash of an image for comparison"""
    image = Image.open(io.BytesIO(image_content))
    return imagehash.average_hash(image)

def download_customer_images():
    """Download customer images from the bucket for matching"""
    customer_hashes = {}
    bucket = storage_client.bucket(CUSTOMER_BUCKET)

    for image_name in customer_images:
        blob = bucket.blob(image_name)
        image_bytes = blob.download_as_bytes()
        customer_hashes[image_name] = get_image_hash(image_bytes)

    return customer_hashes

customer_hashes = download_customer_images()

def match_faces(detected_faces):
    """Match detected faces with stored customer images"""
    matched_customers = set()

    for face in detected_faces:
        face_hash = get_image_hash(cv2.imencode('.jpg', face)[1].tobytes())

        for customer_name, customer_hash in customer_hashes.items():
            if face_hash - customer_hash < 5:  # 5 is the similarity threshold
                matched_customers.add(customer_name)

    return matched_customers

matched_customers = match_faces(faces)
print("Matched Customers:", matched_customers)
```

---

#### **4. Upload Matched Customer Images to Output Bucket**
If a customer appears in the video, upload their corresponding image from the customer base bucket.

```python
OUTPUT_BUCKET = "your-output-bucket"

def upload_matched_customers():
    """Uploads matched customer images to another bucket"""
    output_bucket = storage_client.bucket(OUTPUT_BUCKET)

    for customer in matched_customers:
        blob = storage_client.bucket(CUSTOMER_BUCKET).blob(customer)
        new_blob = output_bucket.blob(customer)
        new_blob.upload_from_string(blob.download_as_bytes())
        print(f"Uploaded {customer} to {OUTPUT_BUCKET}")

upload_matched_customers()
```

---

### **Final Workflow**
1. Extract frames from the video.
2. Detect faces using Google Cloud Vision.
3. Compare detected faces with stored customer images.
4. If a match is found, upload the customer’s image to an output bucket.

This ensures only the **relevant customer images** are uploaded instead of frames from the video. 🚀 Let me know if you need modifications!