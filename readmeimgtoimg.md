gcloud auth login
gcloud auth application-default login
gcloud config set project clpoc1
#gcloud storage buckets add-iam-policy-binding gs://clpoc1-input-video-bucket     --member=serviceAccount:clpoc1svcac1@clpoc1.iam.gserviceaccount.com     --role=roles/storage.objectViewer
cd /c/users/manohar/downloads/Prasannasingle
python videocustbasedetect.py