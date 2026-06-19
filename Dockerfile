FROM python:3.11-slim
LABEL org.opencontainers.image.title="wormhole-sim"
LABEL org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libfreetype6-dev libpng-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN pip install -e .
CMD ["python", "-c", "import core, numerics, visualization; print('wormhole-sim ready')"]
