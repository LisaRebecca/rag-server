# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /knowledge-service

# Copy the service code into the container
COPY . /knowledge-service

# Install dependencies
RUN pip install -r mock-knowledge-service/requirements.txt
RUN pip install -e .

RUN apt update && apt install -y curl

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "mock-knowledge-service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
