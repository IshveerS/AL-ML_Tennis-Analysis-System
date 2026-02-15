from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2


def draw_player_bounding_box(
    frame,
    box: Sequence[float],
    track_id: Optional[int],
    class_name: str,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {track_id}" if track_id is not None else class_name
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_ball_bounding_box(
    frame,
    box: Sequence[float],
    confidence: Optional[float],
    color: Tuple[int, int, int] = (0, 165, 255),
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "ball" if confidence is None else f"ball {confidence:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_player_detections(frame, player_detections: Iterable[Dict]) -> None:
    for detection in player_detections:
        draw_player_bounding_box(
            frame,
            detection.get("box", (0, 0, 0, 0)),
            detection.get("track_id"),
            detection.get("class_name", "player"),
        )


def draw_ball_detections(frame, ball_detections: Iterable[Dict]) -> None:
    for detection in ball_detections:
        draw_ball_bounding_box(
            frame,
            detection.get("box", (0, 0, 0, 0)),
            detection.get("confidence"),
        )
