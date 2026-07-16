FROM python:3.11-slim

ARG REQUIREMENTS_FILE=requirements-gpu.txt
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libheif1 \
    libjpeg62-turbo \
    libturbojpeg0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-gpu.txt ./
COPY vendor/MRZScanner ./vendor/MRZScanner

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" \
       torch==2.11.0 torchvision==0.26.0 \
    && python -m pip install -r "${REQUIREMENTS_FILE}"

# EasyOCR must not download model files after the read-only container starts.
RUN mkdir -p /opt/easyocr \
    && python -c "import easyocr; easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)" \
    && chmod -R a=rX /opt/easyocr

RUN python -c "import capybara"

COPY app ./app
COPY configs ./configs
COPY vendor/rtdetrv2_pytorch ./vendor/rtdetrv2_pytorch


RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/weights /tmp/matplotlib \
    && chown -R app:app /app /tmp/matplotlib

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
