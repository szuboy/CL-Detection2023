#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t cldetection_alg_2023 "$SCRIPTPATH"
