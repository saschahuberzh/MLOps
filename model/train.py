import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "kdkd1/waste-garbage-management-dataset"
KEEP = {"metal", "glass", "biological", "paper", "cardboard", "plastic", "trash"}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
TEST_SIZE = 0.2
SEED = 42
LOG_EVERY_N_BATCHES = 20


def load_raw_dataset(dataset_name: str):
    print(f"[INFO] Lade Datensatz: {dataset_name}")
    ds = load_dataset(dataset_name)["train"]
    print(f"[INFO] Datensatz geladen. Beispiele gesamt: {len(ds)}")
    return ds


def split_dataset(dataset, test_size: float = 0.2, seed: int = 42):
    print(f"[INFO] Splitte Datensatz: test_size={test_size}, seed={seed}")
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    print(f"[INFO] Train-Größe vor Filter: {len(splits['train'])}")
    print(f"[INFO] Val-Größe vor Filter: {len(splits['test'])}")
    return splits["train"], splits["test"]


def get_label_metadata(dataset, keep_labels: set[str]):
    label_names = dataset.features["label"].names
    selected_labels = sorted(list(keep_labels))
    label_to_idx = {name: i for i, name in enumerate(selected_labels)}

    print(f"[INFO] Verfügbare Labels: {label_names}")
    print(f"[INFO] Verwendete Labels: {selected_labels}")

    return label_names, selected_labels, label_to_idx


def filter_dataset(dataset, label_names, keep_labels: set[str], split_name: str = "dataset"):
    print(f"[INFO] Filtere {split_name} nach Labels: {sorted(list(keep_labels))}")

    def keep_example(example):
        return label_names[example["label"]].lower() in keep_labels

    filtered = dataset.filter(keep_example)
    print(f"[INFO] {split_name} nach Filter: {len(filtered)} Beispiele")
    return filtered


def build_transforms(image_size=(224, 224)):
    print(f"[INFO] Erzeuge Transforms mit Bildgröße: {image_size}")

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def add_transforms(dataset, label_names, label_to_idx, image_transform, split_name: str = "dataset"):
    print(f"[INFO] Hänge Transform an {split_name}")

    def transform_batch(batch):
        return {
            "pixel_values": [image_transform(img.convert("RGB")) for img in batch["image"]],
            "labels": [label_to_idx[label_names[y].lower()] for y in batch["label"]],
        }

    return dataset.with_transform(transform_batch)


def prepare_datasets(dataset_name: str, keep_labels: set[str], test_size: float, seed: int):
    print("[INFO] Starte Dataset-Vorbereitung")
    raw_ds = load_raw_dataset(dataset_name)
    train_ds, val_ds = split_dataset(raw_ds, test_size=test_size, seed=seed)

    label_names, selected_labels, label_to_idx = get_label_metadata(raw_ds, keep_labels)

    train_ds = filter_dataset(train_ds, label_names, keep_labels, split_name="train")
    val_ds = filter_dataset(val_ds, label_names, keep_labels, split_name="val")

    train_transform, eval_transform = build_transforms(IMAGE_SIZE)

    train_ds = add_transforms(train_ds, label_names, label_to_idx, train_transform, split_name="train")
    val_ds = add_transforms(val_ds, label_names, label_to_idx, eval_transform, split_name="val")

    print("[INFO] Dataset-Vorbereitung abgeschlossen")
    return train_ds, val_ds, selected_labels


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    return pixel_values, labels


def create_dataloaders(train_ds, val_ds, batch_size: int):
    print(f"[INFO] Erzeuge DataLoader mit batch_size={batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("[INFO] DataLoader erstellt")
    return train_loader, val_loader


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Verwende Device: {device}")
    return device


def build_model(num_classes: int):
    print(f"[INFO] Baue Modell für {num_classes} Klassen")
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_optimizer(model, learning_rate: float):
    print(f"[INFO] Erzeuge Optimizer mit learning_rate={learning_rate}")
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_idx: int):
    model.train()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    print(f"[INFO] Training Epoch {epoch_idx + 1} gestartet")

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        running_loss += loss.item() * labels.size(0)

        if batch_idx % LOG_EVERY_N_BATCHES == 0:
            current_acc = total_correct / total_samples if total_samples > 0 else 0.0
            print(
                f"[INFO] Epoch {epoch_idx + 1} | "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"loss={loss.item():.4f} | acc={current_acc:.4f}"
            )

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"[INFO] Training Epoch {epoch_idx + 1} abgeschlossen | "
        f"avg_loss={avg_loss:.4f} | acc={accuracy:.4f}"
    )

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, epoch_idx: int):
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0

    print(f"[INFO] Evaluation Epoch {epoch_idx + 1} gestartet")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)

            if batch_idx % LOG_EVERY_N_BATCHES == 0:
                current_acc = total_correct / total_samples if total_samples > 0 else 0.0
                print(
                    f"[INFO] Eval Epoch {epoch_idx + 1} | "
                    f"Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"loss={loss.item():.4f} | acc={current_acc:.4f}"
                )

    avg_loss = running_loss / total_samples
    accuracy = total_correct / total_samples

    print(
        f"[INFO] Evaluation Epoch {epoch_idx + 1} abgeschlossen | "
        f"avg_loss={avg_loss:.4f} | acc={accuracy:.4f}"
    )

    return avg_loss, accuracy


def save_artifacts(model, selected_labels, artifact_dir: Path):
    model_path = artifact_dir / "model.pt"
    labels_path = artifact_dir / "labels.json"

    print(f"[INFO] Speichere Modell nach: {model_path}")
    torch.save(model.state_dict(), model_path)

    print(f"[INFO] Speichere Labels nach: {labels_path}")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(selected_labels, f, ensure_ascii=False, indent=2)

    return model_path, labels_path


def run_training_pipeline():
    print("[INFO] Trainingspipeline startet")

    train_ds, val_ds, selected_labels = prepare_datasets(
        dataset_name=DATASET_NAME,
        keep_labels=KEEP,
        test_size=TEST_SIZE,
        seed=SEED,
    )

    train_loader, val_loader = create_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=BATCH_SIZE,
    )

    print("[INFO] Teste ersten Train-Batch")
    first_images, first_labels = next(iter(train_loader))
    print(
        f"[INFO] Erster Batch ok | images.shape={first_images.shape} | "
        f"labels.shape={first_labels.shape}"
    )

    device = get_device()
    model = build_model(num_classes=len(selected_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, LEARNING_RATE)

    best_val_acc = 0.0
    print("[INFO] Starte Training")

    for epoch in range(NUM_EPOCHS):
        print(f"[INFO] ===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch_idx=epoch,
        )

        print(
            f"[INFO] Epoch {epoch + 1} Ergebnis | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"[INFO] Neues bestes Modell gefunden: val_acc={best_val_acc:.4f}")
            model_path, labels_path = save_artifacts(model, selected_labels, ARTIFACT_DIR)
            print(f"[INFO] Modell gespeichert: {model_path}")
            print(f"[INFO] Labels gespeichert: {labels_path}")

    print("[INFO] Training abgeschlossen")


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()