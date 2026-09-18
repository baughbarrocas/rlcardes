FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rlcardes
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.1.0 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .
RUN pip3 install --no-cache-dir -e .

ENTRYPOINT ["python3", "-m", "rlcardes.falconer"]
