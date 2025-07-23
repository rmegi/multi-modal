# Use Python 3.10 base image
FROM python:3.10

# Set working directory
WORKDIR /code


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    git \
    tcpdump \
    iputils-ping \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libcairo2-dev \
    libgirepository1.0-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gtk-3.0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*  

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app/ /code/app
WORKDIR /code/app

# Run your FastAPI app
CMD ["python", "main.py"]
