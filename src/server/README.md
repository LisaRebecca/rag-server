

## FAISS Indices for FAU Data
### Light Index (384 dims, all-MiniLM-L6-v2): 
[Link Text](https://drive.google.com/file/d/1qOECFQ_Df_sBCextiqRbPjeKHTFXpbdW/view?usp=sharing)
### Normal Index (768 dims, all-mpnet-base-v2): 
[Link Text](https://drive.google.com/file/d/1-0ncb5rZ-9SSosAocHnuR6iYIfLLdtNE/view?usp=sharing)

Update `VECTORSTORE_PATH` in the script and run your model.

## How to generate the server certificates
run `sh generate_certs.sh`


##################################

## Testing Queuing on the Web Server
### Step 1: Make the Script Executable (if not already)
First, make the script executable by running the following command in your terminal:

`chmod +x test_queue.sh`

### Step 2: Run the Script
Once the script is executable, run it using:

`sh test_queue.sh`

##################################

## Running The Server Using Docker
### Step 1: Open Docker application on your local machine

### Step 2: From your CMD in the project directory
for first time build run `docker-compose up --scale fastapi-server=3` - Creates 3 instances of the server

run `docker-compose up` or
run `docker-compose up --build`


### Step 3: Finally, to quit the application
run `control c`

##################################

## Running Server Health Tracking System {Prometheus + Grafana}
### Step 1: Run APP using Docker
run `./prometheus --config.file=server/prometheus.yml`

### Step 2: Open browser to "localhost:9090" {Prometheus}
goto `Targets -> Health Tracking`
### Step 3: Open browser to "localhost:3000" {Grafana}
goto `Dashboards`

##################################


