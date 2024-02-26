#!/bin/bash

DOCKER_URI="us-central1-docker.pkg.dev/cloud-nas-260507/pytorch/vllm:gemma"
docker build -f vertex/Dockerfile . -t "${DOCKER_URI}"
