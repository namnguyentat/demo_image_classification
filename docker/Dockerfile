FROM ubuntu:16.04

# Install requirements
RUN apt-get update
RUN apt-get install -y build-essential \
  checkinstall \
  libreadline-gplv2-dev \
  libncursesw5-dev \
  libssl-dev \
  libsqlite3-dev \
  tk-dev \
  libgdbm-dev \
  libc6-dev \
  libbz2-dev \
  zlib1g-dev \
  openssl \
  libffi-dev \
  python3-dev \
  python3-setuptools \
  wget

# Prepare to build
RUN mkdir /tmp/Python37 \
  && cd /tmp/Python37 \
  && wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz \
  && tar xvf Python-3.7.3.tar.xz \
  && cd /tmp/Python37/Python-3.7.3 \
  && ./configure \
  && make altinstall

RUN pip3.7 install --upgrade pip
RUN pip3.7 install imutils \
  matplotlib \
  opencv-python==3.4.5.20 \
  opencv-contrib-python==3.4.5.20 \
  scikit-learn==0.21.2 \
  tensorflow==1.14 \
  keras==2.2.4

ENV APP_HOME /app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME
