FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    fastapi==0.143.0 \
    "uvicorn[standard]==0.54.0" \
    numpy==2.2.6 \
    open_clip_torch==3.3.0 \
    timm==1.0.30 \
    huggingface_hub==2.2.0 \
    safetensors==0.8.0 \
    pillow==12.3.0

WORKDIR /app
COPY server.py ./

EXPOSE 8001

ENTRYPOINT ["python3", "server.py"]
