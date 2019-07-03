#!/bin/bash

set -e

docker-compose -f docker/docker-compose.yml -p demoimageclassification up -d
