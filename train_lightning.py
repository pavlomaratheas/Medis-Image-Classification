import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset import unzip_data, get_dataset
from model_lightning import MedisLightningModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from visualize_results import generate_all_plots


def run_kfold(dataset, k_folds=5, batch_size=16, max_epochs=15):
    # Extraction of labels for stratification
    targets = [label for _, label in dataset.samples]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_labels, all_preds = [], []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Dataset split
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # New model for each fold, so every fold start without precedent weights
        model = MedisLightningModel(n_classes=len(dataset.classes), lr=0.001)

        logger = CSVLogger("logs", name=f"fold_{fold+1}")

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=logger,
            enable_checkpointing=False
        )

        # trainer
        trainer.fit(model, train_loader, val_loader)

        # Prediction on validation set
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())

        # saving results of the fold
        fold_results.append((labels, preds))
        all_labels.extend(labels)
        all_preds.extend(preds)

    # Results
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plots
    print("\n" + "="*60)
    print("Generating visualizations and reports...")
    print("="*60)
    generate_all_plots(
        fold_results=fold_results,
        class_names=dataset.classes,
        log_dir="logs",
        save_dir="plots"
    )

    return fold_results


if __name__ == "__main__":
    unzip_data("Data.zip", "Data")
    dataset = get_dataset("Data", train=True)

    run_kfold(dataset, k_folds=5, batch_size=16, max_epochs=15)