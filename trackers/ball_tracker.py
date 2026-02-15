import os
import pickle
from typing import Dict, List, Optional

from ultralytics import YOLO


class BallTracker:
    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self.model = YOLO(model_path)

    def detect_frames(
        self,
        video_frames: List,
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
        conf: float = 0.2,
    ) -> List[List[Dict]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as handle:
                return pickle.load(handle)

        detections: List[List[Dict]] = []
        for frame in video_frames:
            frame_detections: List[Dict] = []
            results = self.model.predict(frame, conf=conf, verbose=False)
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
                        if class_name not in {"sports ball", "tennis ball", "ball"}:
                            continue
                        confidence = float(box.conf[0]) if box.conf is not None else None
                        xyxy = box.xyxy[0].tolist()
                        frame_detections.append(
                            {
                                "box": xyxy,
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
