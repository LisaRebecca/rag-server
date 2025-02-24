# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /fastapi_RAG_container

# Copy the project structure into the container
COPY . /fastapi_RAG_container

# Install dependencies
RUN pip install -r fastapi_RAG_service/requirements.txt

RUN apt update && apt install -y curl

# Expose the port the application will run on
EXPOSE 8090

# Command to run the application
CMD ["python", "-m", "uvicorn", "fastapi_RAG_service.app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]