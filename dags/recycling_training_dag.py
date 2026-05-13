from datetime import datetime, timezone
from pathlib import Path
import sys

from airflow.decorators import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.train import (
    DATASET_NAME,
    KEEP,
    SEED,
    TEST_SIZE,
    prepare_and_persist_datasets,
    train_from_prepared_data,
)


@dag(
    dag_id="recycling_model_training",
    description="Train the recycling image classifier with Airflow",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["mlops", "training", "recycling"],
)
def recycling_model_training():
    @task(task_id="prepare_dataset")
    def prepare_dataset_task():
        return prepare_and_persist_datasets(
            dataset_name=DATASET_NAME,
            keep_labels=KEEP,
            test_size=TEST_SIZE,
            seed=SEED,
        )

    @task(task_id="train_model")
    def train_model_task(prepared_metadata: dict):
        return train_from_prepared_data(prepared_metadata)

    @task(task_id="summarize_training")
    def summarize_training_task(training_result: dict):
        print("[INFO] Training erfolgreich abgeschlossen")
        print(
            "[INFO] Ergebnis | "
            f"best_val_acc={training_result['best_val_acc']:.4f} | "
            f"model_path={training_result['model_path']} | "
            f"labels_path={training_result['labels_path']} | "
            f"num_classes={training_result['num_classes']}"
        )

    prepared_metadata = prepare_dataset_task()
    training_result = train_model_task(prepared_metadata)
    summarize_training_task(training_result)


recycling_model_training_dag = recycling_model_training()
