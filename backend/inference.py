import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR.parent / "model" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
LABELS_PATH = ARTIFACT_DIR / "labels.json"

print(MODEL_PATH, "THIS IS THE MODEL PATH")

IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"labels.json nicht gefunden: {LABELS_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json ist leer oder ungültig")

    return labels


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class RecyclingPredictor:
    def __init__(self):
        self.labels = load_labels()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"model.pt nicht gefunden: {MODEL_PATH}")

        self.model = build_model(num_classes=len(self.labels))

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

        print(f"[INFO] Model geladen von: {MODEL_PATH}")
        print(f"[INFO] Labels geladen von: {LABELS_PATH}")
        print(f"[INFO] Device: {DEVICE}")
        print(f"[INFO] Klassen: {self.labels}")

    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_label = self.labels[predicted_idx.item()]
        predicted_confidence = float(confidence.item())

        return {
            "label": predicted_label,
            "confidence": predicted_confidence,
        }


predictor = RecyclingPredictor()