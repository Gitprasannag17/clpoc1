Approach 1: Use Google Cloud Console UI (easiest)

Go to Google Cloud AutoML Vision UI
Navigate to your trained model in AutoML Vision.
Click "Test & Use" → Upload Image.
Upload your test image from your local system.
The model will predict and display results.

Approach 3: Upload Images to Cloud Storage & Trigger Prediction (good to have)

Upload test images to a Google Cloud Storage bucket:
```
gsutil cp test_image.jpg gs://your-bucket-name/
```

Modify the Python script to read images from Cloud Storage instead of local disk:
```
from google.cloud import automl_v1beta1 as automl

def predict_image_gcs(project_id, model_id, gcs_image_path):
    client = automl.PredictionServiceClient()
    model_full_id = f"projects/{project_id}/locations/us-central1/models/{model_id}"

    image = automl.types.Image(image_bytes=gcs_image_path)
    payload = automl.types.ExamplePayload(image=image)

    request = automl.types.PredictRequest(name=model_full_id, payload=payload)
    response = client.predict(request=request)

    for result in response.payload:
        print(f"Predicted Person: {result.display_name}, Confidence: {result.classification.score}")

predict_image_gcs("your-project-id", "your-model-id", "gs://your-bucket-name/test_image.jpg")
```


