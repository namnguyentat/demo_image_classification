#!/bin/bash

set -e

if [ "$GPU_ENABLE" == "1" ]; then
  docker-compose -f docker/cpu/docker-compose.yml -p demoimageclassification up -d
else
  docker-compose -f docker/cpu/docker-compose.yml -p demoimageclassification up -d
fi
