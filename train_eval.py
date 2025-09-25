import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dataset import unzip_data, get_dataset
from model import get_model


def cross_validate(model_class, dataset, k_folds=5, epochs=5, batch_size=16, device="cpu"):
    "I use stratified cross-validation with 5 folders"

    targets = [label for _, label in dataset.samples]  # labels for stratification
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_labels, all_preds = [], []
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        # Data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Model, optimizer, loss
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # Training
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        fold_labels, fold_preds = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                fold_labels.extend(targets.cpu().numpy())
                fold_preds.extend(preds.cpu().numpy())

        acc = correct / total
        results[fold] = {"val_loss": val_loss / len(val_loader), "val_acc": acc}
        print(f"Fold {fold+1} - Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {acc:.4f}")

        all_labels.extend(fold_labels)
        all_preds.extend(fold_preds)

        # Save the fold model
        torch.save(model.state_dict(), f"model_fold_{fold+1}.pth")

    # Results
    print("\n Classification Report")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Aggregated across folds)")
    plt.show()

    mean_acc = np.mean([res["val_acc"] for res in results.values()])
    print(f"\nMean Accuracy across folds: {mean_acc:.4f}")

    return results


if __name__ == "__main__":
    unzip_data("Data.zip", "Data")
    dataset = get_dataset("Data", train=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cross_validate(lambda: get_model(n_classes=len(dataset.classes), pretrained=True),
                   dataset, k_folds=5, epochs=5, batch_size=16, device=device)
