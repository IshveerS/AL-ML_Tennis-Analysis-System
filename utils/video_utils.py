import os
from typing import List

import cv2


def read_video(video_path: str) -> List:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()
    return frames


def save_video(output_video_frames: List, output_video_path: str, fps: int = 24) -> None:
    if not output_video_frames:
        raise ValueError("No frames provided to save_video")

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in output_video_frames:
        writer.write(frame)

    writer.release()
