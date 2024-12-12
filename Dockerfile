# Use a base image with both Java and Python support
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Create a working directory for the application
WORKDIR /app

# Copy the requirements file, param file, and other necessary files into the working directory
COPY requirements.txt .
COPY param.py .

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    python3 --version && \
    pip3 --version

# Install dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install OpenJDK
RUN apt-get install -y openjdk-17-jdk && \
    java --version

# Expose ports for both services
EXPOSE 8000 8080

# Command to run both services
CMD ["bash", "-c", "uvicorn param:app --host 0.0.0.0 --port 8000"]
