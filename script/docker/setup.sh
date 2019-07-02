#!/bin/bash

# if ! [[ $(pwd) =~ /venus/venus-api$ ]]; then
#   echo "ERROR: Wrong directory. Please change directory to /venus/venus-api and try again"
#   exit 1
# fi

set -e

docker-compose -f docker/docker-compose.yml -p demoimageclassification build
