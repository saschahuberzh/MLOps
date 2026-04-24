# Setup
conda create -n recycling-ai python=3.10 -y  
conda activate recycling-ai  
pip list  
conda install pip -y  
pip install -r requirements.txt  
pip list

# DVC / GCP Setup
brew install --cask google-cloud-sdk  
gcloud init  
gcloud auth application-default login

(Einmalig) Projekt erstellt in gcp: mlops-zhaw  
gcloud config set project mlops-zhaw-494207  
gcloud config get-value project  
gcloud storage buckets create gs://recycling-ai-mlops-sascha --location=europe-west6

dvc remote add -d storage gs://recycling-ai-mlops-sascha/recycling-dvc.  
dvc remote list

# For development

## Run frontend (only for development)
cd frontend  
streamlit run frontend.py

## Run backend (only for development)
cd backend  
uvicorn backend:app --reload

# Run frontend and backend in docker
docker compose -f docker-compose.local.yml up --build

# Dataset
Dataset: https://huggingface.co/datasets/kdkd1/waste-garbage-management-dataset  
Alternative Dataset: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

# Model training

cd model  
python train.py

Wenn model abgespeichert werden soll:  
dvc push  
git add ../.dvc/config  
git add artifacts/model.pt.dvc  
git commit -m "add model v1 with dvc tracking"  
git tag model-v1  
git push --tags  

# GCP Deployment
APIs aktivieren:  
gcloud services enable artifactregistry.googleapis.com iamcredentials.googleapis.com  

Docker repository in Artifact registry erstellen:  
gcloud artifacts repositories create recycling-ai \
  --repository-format=docker \
  --location=europe-west6 \
  --description="Docker images for backend and frontend"

Überprüfen: 
gcloud artifacts repositories list --location=europe-west6

(Einmalig) Service Account erstellen:  
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

Zugriff auf artifact registry geben:  
gcloud projects add-iam-policy-binding mlops-zhaw-494207 \
  --member="serviceAccount:github-actions@mlops-zhaw-494207.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

Projekt nummer erhalen: //93698602367  
gcloud projects describe mlops-zhaw-494207 --format="value(projectNumber)"

Workload Identity Pool anlegen:  
gcloud iam workload-identity-pools create github \
  --project=mlops-zhaw-494207 \
  --location=global \
  --display-name="GitHub Pool"

OIDC-Provider für GitHub anlegen:  
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --project=mlops-zhaw-494207 \
  --location=global \
  --workload-identity-pool=github \
  --display-name="GitHub Provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'saschahuberzh'"

Überprüfen:  
gcloud iam workload-identity-pools providers describe github-provider \
  --project=mlops-zhaw-494207 \
  --location=global \
  --workload-identity-pool=github

Binding befehl:  
  gcloud iam service-accounts add-iam-policy-binding \
  github-actions@mlops-zhaw-494207.iam.gserviceaccount.com \
  --project=mlops-zhaw-494207 \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/93698602367/locations/global/workloadIdentityPools/github/attribute.repository/saschahuberzh/MLOps"

Provider namem für github secrets holen: //projects/93698602367/locations/global/workloadIdentityPools/github/providers/github-provider  
gcloud iam workload-identity-pools providers describe github-provider \
  --project=mlops-zhaw-494207 \
  --location=global \
  --workload-identity-pool=github \
  --format="value(name)"

GitHub → Settings → Secrets and variables → Actions

Secrets
GCP_WORKLOAD_IDENTITY_PROVIDER
projects/93698602367/locations/global/workloadIdentityPools/github/providers/github-provider
GCP_SERVICE_ACCOUNT
github-actions@mlops-zhaw-494207.iam.gserviceaccount.com

Secrets
GCP_WORKLOAD_IDENTITY_PROVIDER
projects/93698602367/locations/global/workloadIdentityPools/github/providers/github-provider
GCP_SERVICE_ACCOUNT
github-actions@mlops-zhaw-494207.iam.gserviceaccount.com

Cloud run api aktivieren:  
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

Run backend:
gcloud run deploy recycling-backend \
  --image=europe-west6-docker.pkg.dev/mlops-zhaw-494207/recycling-ai/backend:latest \
  --region=europe-west6 \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000

Run frontend:
gcloud run deploy recycling-frontend \
  --image=europe-west6-docker.pkg.dev/mlops-zhaw-494207/recycling-ai/frontend:latest \
  --region=europe-west6 \
  --platform=managed \
  --allow-unauthenticated \
  --port=8501 \
  --set-env-vars BACKEND_PREDICT_URL=https://recycling-backend-93698602367.europe-west6.run.app/predict

Delete running containers:
gcloud run services delete recycling-backend --region=europe-west6
gcloud run services delete recycling-frontend --region=europe-west6

# Remove
conda deactivate  
conda remove -n recycling-ai --all    