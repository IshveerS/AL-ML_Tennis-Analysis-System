# AI ML Tennis Analysis System

## Overview
Starter scaffold for a tennis analysis pipeline using YOLO, PyTorch, and court key point detection.

## Project Layout
- input_videos: Raw videos for analysis
- output_videos: Rendered outputs
- models: Trained model weights
- utils: Video IO and drawing helpers
- trackers: Player tracking utilities
- tracker_stubs: Cached detections
- court_line_detector: Court key point detection
- mini_court: Mini-court visualization
- notebooks: Training notebooks

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   - opencv-python
   - ultralytics
   - torch
   - torchvision
   - numpy
3. Place input video at input_videos/input_video.mp4

## Run
Execute main.py to process the video and write output to output_videos/output_video.avi

## Notes
- The key point model expects weights at models/key_points_model.pth
- The player tracker can load cached detections from tracker_stubs/player_detections.pkl
