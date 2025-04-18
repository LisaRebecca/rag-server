# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /fau_rag

# Copy the project structure into the container
COPY . /fau_rag
COPY ./rag_service/entrypoint.sh /entrypoint.sh

# Install dependencies
RUN pip install -r rag_service/requirements.txt

RUN apt update && apt install -y curl

# Expose the port the application will run on
EXPOSE 8090

# Command to run the application
CMD ["/entrypoint.sh"]
