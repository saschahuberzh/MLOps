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

# Remove
conda deactivate
conda remove -n recycling-ai --all