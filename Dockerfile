# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and model
COPY app_fastapi.py .
COPY iris_model.pkl .

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]