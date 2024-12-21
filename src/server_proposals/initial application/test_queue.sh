#!/bin/bash

# Number of concurrent requests
num_requests=100

for i in $(seq 1 $num_requests); do
  curl -X POST -H "Content-Type: application/json" \
    -d '{"prompt": "This is a test prompt"}' \
    --insecure \
    https://127.0.0.1:8080/generate-response &
done

wait
