FROM python:3.12

# set work directory
WORKDIR /usr/srv

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#Install  TensorFlow Lite model with root permission (mediapipe requirment)
RUN mkdir -p /usr/local/lib/python3.12/site-packages/mediapipe/modules/pose_landmark/ && \     
    chmod -R 777 /usr/local/lib/python3.12/site-packages/mediapipe/modules/

RUN useradd -rm -d /code -s /bin/bash -g root -G sudo -u 1001 ubuntu

# Install system dependencies for openCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

# copy requirements file
COPY ./requirements.txt /usr/srv/requirements.txt
COPY ./setup.py /usr/srv/setup.py
RUN apt-get install libpq-dev
RUN pip install --no-cache-dir --upgrade -r requirements.txt
#rendere un package
RUN pip install -e .  

USER ubuntu

EXPOSE 8000

CMD bash -c 'uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'