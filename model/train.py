import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset, load_from_disk
import wandb


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DATA_DIR = ARTIFACT_DIR / "prepared_dataset"

DATASET_NAME = "kdkd1/waste-garbage-management-dataset"
KEEP = {"metal", "glass", "biological", "paper", "battery", "trash", "cardboard", "shoes", "clothes", "plastic"}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
TEST_SIZE = 0.2
SEED = 42
LOG_EVERY_N_BATCHES = 20


def get_wandb_mode():
    if os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return "disabled"

    if os.getenv("WANDB_API_KEY"):
        return os.getenv("WANDB_MODE", "online")

    return os.getenv("WANDB_MODE", "offline")


def init_wandb_run(selected_labels):
    mode = get_wandb_mode()
    if mode == "disabled":
        print("[INFO] W&B ist deaktiviert oder nicht konfiguriert")
        return None

    project_name = os.getenv("WANDB_PROJECT", "recycling-airflow-training")
    run_name = os.getenv("WANDB_RUN_NAME")

    print(f"[INFO] Starte W&B Run | project={project_name} | mode={mode}")
    return wandb.init(
        project=project_name,
        name=run_name,
        mode=mode,
        config={
            "dataset_name": DATASET_NAME,
            "keep_labels": sorted(list(KEEP)),
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "test_size": TEST_SIZE,
            "seed": SEED,
            "num_classes": len(selected_labels),
        },
    )


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


def prepare_and_persist_datasets(dataset_name: str, keep_labels: set[str], test_size: float, seed: int):
    print("[INFO] Starte persistente Dataset-Vorbereitung")
    raw_ds = load_raw_dataset(dataset_name)
    train_ds, val_ds = split_dataset(raw_ds, test_size=test_size, seed=seed)

    label_names, selected_labels, _ = get_label_metadata(raw_ds, keep_labels)

    train_ds = filter_dataset(train_ds, label_names, keep_labels, split_name="train")
    val_ds = filter_dataset(val_ds, label_names, keep_labels, split_name="val")

    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PREPARED_DATA_DIR / "train"
    val_path = PREPARED_DATA_DIR / "val"
    metadata_path = PREPARED_DATA_DIR / "metadata.json"

    print(f"[INFO] Speichere vorbereiteten Train-Split nach: {train_path}")
    train_ds.save_to_disk(str(train_path))

    print(f"[INFO] Speichere vorbereiteten Val-Split nach: {val_path}")
    val_ds.save_to_disk(str(val_path))

    metadata = {
        "dataset_name": dataset_name,
        "selected_labels": selected_labels,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    print(f"[INFO] Speichere Dataset-Metadaten nach: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def load_prepared_datasets(train_path: str, val_path: str, selected_labels: list[str]):
    print(f"[INFO] Lade vorbereiteten Train-Split aus: {train_path}")
    train_ds = load_from_disk(train_path)
    print(f"[INFO] Lade vorbereiteten Val-Split aus: {val_path}")
    val_ds = load_from_disk(val_path)

    label_names = train_ds.features["label"].names
    label_to_idx = {name: i for i, name in enumerate(selected_labels)}

    train_transform, eval_transform = build_transforms(IMAGE_SIZE)
    train_ds = add_transforms(train_ds, label_names, label_to_idx, train_transform, split_name="train")
    val_ds = add_transforms(val_ds, label_names, label_to_idx, eval_transform, split_name="val")

    return train_ds, val_ds


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


def train_from_prepared_data(prepared_metadata: dict):
    print("[INFO] Training aus vorbereiteten Daten startet")

    selected_labels = prepared_metadata["selected_labels"]
    train_path = prepared_metadata["train_path"]
    val_path = prepared_metadata["val_path"]

    train_ds, val_ds = load_prepared_datasets(
        train_path=train_path,
        val_path=val_path,
        selected_labels=selected_labels,
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

    wandb_run = init_wandb_run(selected_labels)

    best_val_acc = float("-inf")
    best_model_path = None
    best_labels_path = None
    print("[INFO] Starte Training")

    try:
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
                best_model_path = model_path
                best_labels_path = labels_path
                print(f"[INFO] Modell gespeichert: {model_path}")
                print(f"[INFO] Labels gespeichert: {labels_path}")

            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "best_val_accuracy": best_val_acc,
                    }
                )

        print("[INFO] Training abgeschlossen")

        if best_model_path is None or best_labels_path is None:
            raise RuntimeError("Training abgeschlossen, aber keine Artefakte gespeichert wurden")

        if wandb_run is not None:
            wandb_run.summary["best_val_accuracy"] = best_val_acc
            wandb_run.summary["model_path"] = str(best_model_path)
            wandb_run.summary["labels_path"] = str(best_labels_path)

        return {
            "best_val_acc": best_val_acc,
            "model_path": str(best_model_path),
            "labels_path": str(best_labels_path),
            "num_classes": len(selected_labels),
        }
    finally:
        if wandb_run is not None:
            wandb.finish()


def run_training_pipeline():
    print("[INFO] Trainingspipeline startet")
    prepared_metadata = prepare_and_persist_datasets(
        dataset_name=DATASET_NAME,
        keep_labels=KEEP,
        test_size=TEST_SIZE,
        seed=SEED,
    )
    return train_from_prepared_data(prepared_metadata)


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()

    #asdf