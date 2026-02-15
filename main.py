import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from utils import read_video, save_video, draw_ball_detections, draw_player_detections
from trackers import BallTracker, PlayerTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt


def _box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _meters_per_pixel(key_points: Sequence[Tuple[float, float]]) -> Optional[float]:
    if not key_points:
        return None
    xs = [point[0] for point in key_points]
    ys = [point[1] for point in key_points]
    width_px = max(xs) - min(xs)
    if width_px <= 0:
        return None
    court_width_m = 10.97
    return court_width_m / width_px


def _court_bounds(key_points: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not key_points:
        return None
    xs = [point[0] for point in key_points]
    ys = [point[1] for point in key_points]
    return min(xs), min(ys), max(xs), max(ys)


def _best_ball_detection(detections: List[Dict]) -> Optional[Dict]:
    if not detections:
        return None
    return max(detections, key=lambda det: det.get("confidence", 0) or 0)


def _overlay_metrics(
    frame,
    shot_count: int,
    ball_speed: Optional[float],
    ball_in: Optional[bool],
    player_speeds: Dict[int, float],
    player_distances: Dict[int, float],
    speed_unit: str,
    distance_unit: str,
) -> None:
    y = 25
    cv2.putText(frame, f"Shots: {shot_count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    if ball_speed is not None:
        cv2.putText(
            frame,
            f"Ball: {ball_speed:.1f} {speed_unit}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 25
    if ball_in is not None:
        status = "IN" if ball_in else "OUT"
        cv2.putText(frame, f"Ball: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    for track_id, speed in player_speeds.items():
        distance = player_distances.get(track_id, 0.0)
        cv2.putText(
            frame,
            f"P{track_id}: {speed:.1f} {speed_unit}, {distance:.1f} {distance_unit}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 25


def main() -> None:
    input_video_path = os.path.join("input_videos", "input_video.mp4")
    output_video_path = os.path.join("output_videos", "output_video.avi")
    os.makedirs("output_videos", exist_ok=True)

    video_frames = read_video(input_video_path)
    frame_skip = 3
    video_frames = video_frames[::frame_skip]
    base_fps = 24
    effective_fps = base_fps / frame_skip if frame_skip else base_fps

    player_tracker = PlayerTracker(model_path="yolov8n.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join("tracker_stubs", "player_detections.pkl"),
    )
    player_detections = PlayerTracker.filter_players(player_detections)

    ball_tracker = BallTracker(model_path="yolov8n.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=True,
        stub_path=os.path.join("tracker_stubs", "ball_detections.pkl"),
        conf=0.25,
    )

    key_points_model_path = os.path.join("models", "key_points_model.pth")
    key_points = []
    if os.path.exists(key_points_model_path) and video_frames:
        court_detector = CourtLineDetector(model_path=key_points_model_path)
        key_points = court_detector.predict(video_frames[0])
        if key_points:
            court_detector.draw_key_points_on_video(video_frames, key_points)

    mini_court = MiniCourt(key_points) if key_points else None
    meters_per_pixel = _meters_per_pixel(key_points)
    court_bounds = _court_bounds(key_points)
    speed_unit = "km/h" if meters_per_pixel else "px/s"
    distance_unit = "m" if meters_per_pixel else "px"

    last_ball_center = None
    last_shot_frame = -999
    shot_count = 0
    player_last_pos: Dict[int, Tuple[float, float]] = {}
    player_distances: Dict[int, float] = {}

    for frame_index, frame in enumerate(video_frames):
        detections = player_detections[frame_index] if frame_index < len(player_detections) else []
        balls = ball_detections[frame_index] if frame_index < len(ball_detections) else []

        draw_player_detections(frame, detections)
        draw_ball_detections(frame, balls)

        if mini_court:
            mini_court.draw_mini_court(frame)
            for detection in detections:
                pos = mini_court.map_player_position_to_mini_court(detection["box"])
                if pos:
                    mini_court.draw_player_on_mini_court(frame, pos)

        best_ball = _best_ball_detection(balls)
        ball_speed = None
        ball_in = None
        ball_center = None
        if best_ball:
            ball_center = _box_center(best_ball["box"])
            if last_ball_center is not None:
                dx = ball_center[0] - last_ball_center[0]
                dy = ball_center[1] - last_ball_center[1]
                dist_px = (dx * dx + dy * dy) ** 0.5
                if meters_per_pixel:
                    speed_mps = dist_px * meters_per_pixel * effective_fps
                    ball_speed = speed_mps * 3.6
                else:
                    ball_speed = dist_px * effective_fps
            last_ball_center = ball_center

            if mini_court:
                mini_ball_pos = mini_court.map_player_position_to_mini_court(
                    (ball_center[0], ball_center[1], ball_center[0] + 1, ball_center[1] + 1)
                )
                if mini_ball_pos:
                    mini_court.draw_player_on_mini_court(frame, mini_ball_pos, color=(0, 165, 255))

            if court_bounds:
                min_x, min_y, max_x, max_y = court_bounds
                ball_in = min_x <= ball_center[0] <= max_x and min_y <= ball_center[1] <= max_y

            for detection in detections:
                center = _box_center(detection["box"])
                dx = center[0] - ball_center[0]
                dy = center[1] - ball_center[1]
                distance = (dx * dx + dy * dy) ** 0.5
                width = detection["box"][2] - detection["box"][0]
                height = detection["box"][3] - detection["box"][1]
                threshold = max(width, height) * 0.6
                if distance < threshold and frame_index - last_shot_frame > 10:
                    shot_count += 1
                    last_shot_frame = frame_index
                    break

        player_speeds: Dict[int, float] = {}
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            center = _box_center(detection["box"])
            last_pos = player_last_pos.get(track_id)
            if last_pos is not None:
                dx = center[0] - last_pos[0]
                dy = center[1] - last_pos[1]
                dist_px = (dx * dx + dy * dy) ** 0.5
                if meters_per_pixel:
                    dist_m = dist_px * meters_per_pixel
                    player_distances[track_id] = player_distances.get(track_id, 0.0) + dist_m
                    speed_mps = dist_m * effective_fps
                    player_speeds[track_id] = speed_mps * 3.6
                else:
                    player_distances[track_id] = player_distances.get(track_id, 0.0) + dist_px
                    player_speeds[track_id] = dist_px * effective_fps
            player_last_pos[track_id] = center

        _overlay_metrics(
            frame,
            shot_count=shot_count,
            ball_speed=ball_speed,
            ball_in=ball_in,
            player_speeds=player_speeds,
            player_distances=player_distances,
            speed_unit=speed_unit,
            distance_unit=distance_unit,
        )

    save_video(video_frames, output_video_path)


if __name__ == "__main__":
    main()
