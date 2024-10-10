# Dockerfile for StrusiNet2

# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary dependencies directly in the Dockerfile
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    pandas \
    scikit-learn \
    numpy

# Copy the rest of the application code
COPY src/ ./src/
COPY example_data/ ./example_data/
COPY saved_model/ ./saved_model/
COPY tests/ ./tests/

# Set an environment variable to avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set an environment variable to avoid buffering python stdout and stderr
ENV PYTHONUNBUFFERED=1

# Specify the entry point command (can be overridden if necessary)
ENTRYPOINT ["python", "src/main.py"]

# Example usage for running the embedding generation tool
# CMD ["--dot_bracket", "((..))((..))", "--model_path", "saved_model/siamese_trained_model.pth"]