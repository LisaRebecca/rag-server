
## Run (from Python code)
Make sure you installed the requirements (`pip install -r requirements.txt`) and the whole service category root folder (when youre in `knowledge-service/` with `pip install -e . `).

Now you can run this compounded command to stay in the root dir (`knowledge-service/`) (staying in this dir all the time is mostly the best option):

`cd mock-knowledge-service/app; uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

Alternatively (not recommended) you can go in this directory: 

`cd mock-knowledge-service/app` 

then you can execute the main script:

`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`


## Build
You should be in this service category's root directory  `knowledge-service/`

`docker build -f mock-knowledge-service/Dockerfile -t mock-knowledge-service .`

## Run in Container

Runnable from anywhere on your machine (remember to start the docker daemon).

Without terminal: `docker run -d -p 8000:8000 mock-knowledge-service`

With connected terminal: `docker run -it --entrypoint /bin/bash -p 8000:8000 mock-knowledge-service`

