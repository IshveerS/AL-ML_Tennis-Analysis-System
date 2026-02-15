import os
import pickle
from typing import Dict, List, Optional

from ultralytics import YOLO


class PlayerTracker:
    def __init__(self, model_path: str = "yolov8x.pt") -> None:
        self.model = YOLO(model_path)

    def detect_frames(
        self,
        video_frames: List,
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[List[Dict]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as handle:
                return pickle.load(handle)

        detections: List[List[Dict]] = []
        for frame in video_frames:
            frame_detections: List[Dict] = []
            results = self.model.track(frame, persist=True)
            if results:
                result = results[0]
                names = result.names
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                        class_name = (
                            names.get(cls_id, str(cls_id))
                            if isinstance(names, dict)
                            else names[cls_id]
                        )
                        if class_name not in {"person", "player"}:
                            continue
                        track_id = int(box.id[0]) if box.id is not None else None
                        confidence = float(box.conf[0]) if box.conf is not None else None
                        xyxy = box.xyxy[0].tolist()
                        frame_detections.append(
                            {
                                "box": xyxy,
                                "track_id": track_id,
                                "class_name": class_name,
                                "confidence": confidence,
                            }
                        )
            detections.append(frame_detections)

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as handle:
                pickle.dump(detections, handle)

        return detections

    @staticmethod
    def filter_players(player_detections: List[List[Dict]], max_players: int = 2) -> List[List[Dict]]:
        filtered: List[List[Dict]] = []
        for frame_detections in player_detections:
            if len(frame_detections) <= max_players:
                filtered.append(frame_detections)
                continue

            frame_detections = sorted(
                frame_detections,
                key=lambda det: (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1]),
                reverse=True,
            )
            filtered.append(frame_detections[:max_players])

        return filtered
