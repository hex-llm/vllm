# Dockerfile for vLLM serving.
# It requires at least an n1-highmem-16 machine to build.
# To build:
# docker build -f model_oss/vllm/dockerfile/serve.Dockerfile . -t ${YOUR_IMAGE_TAG}
#
# To push to gcr:
# docker tag ${YOUR_IMAGE_TAG} gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}
# docker push gcr.io/{YOUR_PROJECT}/${YOUR_IMAGE_TAG}

# The base image is required by vllm:
# https://vllm.readthedocs.io/en/latest/getting_started/installation.html
# Refer to the nvcr docker hub for the full list:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.12-py3

USER root

# Install tools.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends wget
RUN apt-get install -y --no-install-recommends git

# Copy license.
RUN wget https://raw.githubusercontent.com/GoogleCloudPlatform/vertex-ai-samples/main/LICENSE

# Install libraries.
ENV PIP_ROOT_USER_ACTION=ignore
RUN python -m pip install --upgrade pip
RUN pip install google-cloud-storage==2.7.0
RUN pip install absl-py==1.4.0
RUN pip install boto3==1.26.9

# Install vllm deps.
RUN pip install ninja==1.11.1
RUN pip install psutil==5.9.5
RUN pip install ray==2.7.0
RUN pip install sentencepiece==0.1.99
RUN pip install fastapi==0.100.1
RUN pip install uvicorn[standard]==0.23.2
RUN pip install pydantic==1.10.12
RUN pip install --upgrade torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --upgrade xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
RUN pip install packaging==23.2
RUN pip install transformers==4.38.0

COPY vertex/vllm_startup_prober.sh /model_garden/scripts/vllm_startup_prober.sh

# Install vllm from source.
COPY . /workspace/vllm
WORKDIR /workspace/vllm
RUN pip install -e . -v

# Expose port 7080 for host serving.
EXPOSE 7080
