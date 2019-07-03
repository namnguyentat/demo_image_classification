# FROM tensorflow/tensorflow:latest-gpu-py3
FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && \
  apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  x11vnc \
  xvfb \
  xterm \
  fluxbox && \
  apt-get clean

# Create and configure the VNC user
ARG VNCPASS
ENV VNCPASS ${VNCPASS:-secret}

RUN useradd remote --create-home --shell /bin/bash --user-group --groups adm,sudo && \
  echo "remote:$VNCPASS" | chpasswd

# Install deep-learning libs
RUN pip install imutils \
  matplotlib \
  opencv-python==3.4.5.20 \
  opencv-contrib-python==3.4.5.20 \
  scikit-learn==0.21.2 \
  keras==2.2.4

ENV APP_HOME /app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

EXPOSE 5900
