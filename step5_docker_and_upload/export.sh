#!/usr/bin/env bash

./build.sh

docker save cldetection_alg_2023 | gzip -c > CLdetection_Alg_2023.tar.gz
