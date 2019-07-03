#!/bin/bash

set -e

if [ "$GPU_ENABLE" == "1" ]
then
  docker-compose -f docker/gpu/docker-compose.yml -p demoimageclassification run --rm --entrypoint=/bin/bash main -c "$@"
else
  docker-compose -f docker/cpu/docker-compose.yml -p demoimageclassification run --rm --entrypoint=/bin/bash main -c "$@"
fi
