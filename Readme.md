# Setup
conda create -n recycling-ai python=3.10 -y

conda activate recycling-aipip list

conda install pip -y

pip install -r requirements.txt

pip list

# For development

## Run frontend (only for development)
cd frontend
streamlit run frontend.py

## Run backend (only for development)
cd backend
uvicorn backend:app --reload

# Run frontend and backend in docker
docker compose up --build

# Information
Dataset: https://huggingface.co/datasets/kdkd1/waste-garbage-management-dataset
Alternative Dataset: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

# DVC / GCP
brew install --cask google-cloud-sdk
gcloud init
gcloud auth application-default login

dvc add model/artifacts/model.pt

Projekt erstellt in gcp: mlops-zhaw
gcloud config set project mlops-zhaw-494207
gcloud config get-value project
gcloud storage buckets create gs://recycling-ai-mlops-sascha --location=europe-west6

dvc remote add -d storage gs://recycling-ai-mlops-sascha/recycling-dvc
dvc remote list
dvc push

todo: gitignore model

# Remove
conda deactivate
conda remove -n recycling-ai --all