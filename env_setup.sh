#!/bin/bash

ROOT_DIR=`pwd`
export PYTHONPATH=${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}

PATH=${ROOT_DIR}/Common:${PATH}

SSD_DIR=${ROOT_DIR}/../ssd_detectors
export PYTHONPATH=${PYTHONPATH}:${SSD_DIR}

# export LIBRARY_PATH=/usr/local/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
