#!/bin/bash
set -e
WORKSPACE=./
trap 'echo "Error: command failed: $BASH_COMMAND"; exit 1' ERR
conda create -n tranxadapter python=3.10 -y
conda activate tranxadapter
pip install -e .
cd llavanpr
pip install -e .
cd ${WORKSPACE}/qwen3vlnpr
pip install -e .
cd ${WORKSPACE}/ms-swift
pip install -e .
cd ${WORKSPACE}/VLMEvalKit
pip install -e .
cd $WORKSPACE
echo "Conda environment name tranxadapter has been created🎉. Now you can run "conda activate tranxadapter""
