FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy gradio

# Expose port 
EXPOSE 4000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "4000"]