FROM python:3.11

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend

WORKDIR /app/backend

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]