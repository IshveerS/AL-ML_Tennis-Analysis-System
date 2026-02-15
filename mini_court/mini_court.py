from typing import Iterable, List, Sequence, Tuple

import cv2


class MiniCourt:
    def __init__(self, court_key_points: Sequence[Tuple[float, float]], width: int = 300, height: int = 160) -> None:
        self.court_key_points = list(court_key_points)
        self.width = width
        self.height = height

    def draw_mini_court(self, frame) -> None:
        overlay = frame.copy()
        start_x, start_y = 20, 20
        end_x, end_y = start_x + self.width, start_y + self.height

        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        mid_x = start_x + self.width // 2
        cv2.line(overlay, (mid_x, start_y), (mid_x, end_y), (255, 255, 255), 1)
        frame[:, :] = overlay

    def map_player_position_to_mini_court(
        self,
        player_bbox: Sequence[float],
        court_key_points: Sequence[Tuple[float, float]] | None = None,
    ) -> Tuple[int, int] | None:
        key_points = list(court_key_points) if court_key_points is not None else self.court_key_points
        if not key_points:
            return None

        xs = [pt[0] for pt in key_points]
        ys = [pt[1] for pt in key_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x == min_x or max_y == min_y:
            return None

        x1, y1, x2, y2 = player_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        norm_x = (center_x - min_x) / (max_x - min_x)
        norm_y = (center_y - min_y) / (max_y - min_y)

        mini_x = int(20 + norm_x * self.width)
        mini_y = int(20 + norm_y * self.height)
        return mini_x, mini_y

    @staticmethod
    def draw_player_on_mini_court(frame, player_pos: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 255)) -> None:
        cv2.circle(frame, player_pos, 4, color, -1)
