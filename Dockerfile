# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Copy your local project files into the Docker image
COPY . .

# Install python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create an uploads folder so the server can save images temporarily
RUN mkdir -p uploads && chmod 777 uploads

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the Flask app on port 7860
CMD ["python", "app.py"]
