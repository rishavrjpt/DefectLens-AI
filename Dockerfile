FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.cache/torch \
    MPLCONFIGDIR=/tmp/matplotlib

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-download ImageNet ResNet-18 weights so startup needs no internet
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY data/dataset_summary.json ./data/dataset_summary.json
COPY models ./models

COPY results ./results
RUN mkdir -p results/plots && chown -R user:user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
