from typing import Iterable, List, Sequence, Tuple

import cv2
import torch
from torchvision import models, transforms


class CourtLineDetector:
    def __init__(self, model_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _build_model() -> torch.nn.Module:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, 28)
        return model

    def predict(self, image) -> List[Tuple[float, float]]:
        if image is None:
            return []

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor).squeeze(0).cpu()

        points = outputs.view(14, 2).clamp(0, 1)
        denormalized = [(float(x * width), float(y * height)) for x, y in points]
        return denormalized

    @staticmethod
    def draw_key_points(frame, key_points: Sequence[Tuple[float, float]]) -> None:
        for x, y in key_points:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    def draw_key_points_on_video(self, video_frames: Iterable, key_points: Sequence[Tuple[float, float]]) -> None:
        for frame in video_frames:
            self.draw_key_points(frame, key_points)
