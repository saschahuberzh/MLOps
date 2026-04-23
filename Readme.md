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
docker compose up --build

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

# Remove
conda deactivate  
conda remove -n recycling-ai --all