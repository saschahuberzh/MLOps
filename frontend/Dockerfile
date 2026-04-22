FROM python:3.11

WORKDIR /app

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend /app/frontend

WORKDIR /app/frontend

CMD ["streamlit", "run", "frontend.py", "--server.address", "0.0.0.0", "--server.port", "8501"]