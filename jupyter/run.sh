#!/bin/bash
#
# Set ups a Jupyter Lab server with the current 
# directory as the notebook directory.
# See: https://jupyter-docker-stacks.readthedocs.io/en/latest/
#
# Author: Jason Cox
# Date: 2024-02-190
# https://github.com/jasonacox/ProtosAI

echo "Removing old container..."
docker stop jupyter
docker rm jupyter

echo "Starting Jupyter Hub service..."
docker run -d  \
        --shm-size=10.24gb \
        --gpus all \
        -p 8888:8888 \
        -e JUPYTER_ENABLE_LAB=yes \
        -v "${PWD}":/home/jovyan/work \
        --name jupyter \
        quay.io/jupyter/datascience-notebook:2024-01-15 start-notebook.sh --NotebookApp.token='' --notebook-dir=/home/jovyan/work

echo ""
echo "Jupyter Hub is running on http://localhost:8888/lab"
echo ""
echo "Tailing logs.... ^C to exit"
docker logs -f jupyter
