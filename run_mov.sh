#!/usr/bin/env bash

python3 camtrack/camtrack.py "dataset/$1/rgb.mov" "dataset/$1/camera.yml" "dataset/$1/track.txt" "dataset/$1/cloud.txt"
python3 camtrack/cmptrack.py "dataset/$1/ground_truth.yml" "dataset/$1/track.txt" -p