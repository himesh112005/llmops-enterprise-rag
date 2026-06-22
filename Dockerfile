# backend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Requirements copy karke install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saara code copy karein
COPY . .

# FastAPI ka port expose karein
EXPOSE 8000

# Server start karne ka command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]