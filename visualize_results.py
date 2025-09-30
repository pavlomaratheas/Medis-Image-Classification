import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import os

def plot_average_metrics(log_dir="logs", save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)

    fold_dirs = sorted([d for d in Path(log_dir).iterdir() if d.is_dir() and d.name.startswith("fold_")])

    if not fold_dirs:
        return

    # Collect metrics from all folds
    all_train_loss, all_val_loss = [], []
    all_train_acc, all_val_acc = [], []

    max_len = 0

    for fold_dir in fold_dirs:
        version_dirs = list(fold_dir.glob("version_*"))
        if not version_dirs:
            continue

        metrics_file = version_dirs[0] / "metrics.csv"
        if not metrics_file.exists():
            continue

        df = pd.read_csv(metrics_file)

        train_loss = df['train_loss'].dropna().values
        val_loss = df['val_loss'].dropna().values
        train_acc = df['train_acc'].dropna().values
        val_acc = df['val_acc'].dropna().values

        max_len = max(max_len, len(train_loss), len(val_loss), len(train_acc), len(val_acc))

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_acc.append(train_acc)
        all_val_acc.append(val_acc)

    # Pad arrays to same length
    def pad_arrays(arrays, target_len):
        padded = []
        for arr in arrays:
            if len(arr) < target_len:
                padded.append(np.pad(arr, (0, target_len - len(arr)), mode='edge'))
            else:
                padded.append(arr[:target_len])
        return np.array(padded)

    all_train_loss = pad_arrays(all_train_loss, max_len)
    all_val_loss = pad_arrays(all_val_loss, max_len)
    all_train_acc = pad_arrays(all_train_acc, max_len)
    all_val_acc = pad_arrays(all_val_acc, max_len)

    # Calculate mean and std
    mean_train_loss = np.mean(all_train_loss, axis=0)
    std_train_loss = np.std(all_train_loss, axis=0)
    mean_val_loss = np.mean(all_val_loss, axis=0)
    std_val_loss = np.std(all_val_loss, axis=0)

    mean_train_acc = np.mean(all_train_acc, axis=0)
    std_train_acc = np.std(all_train_acc, axis=0)
    mean_val_acc = np.mean(all_val_acc, axis=0)
    std_val_acc = np.std(all_val_acc, axis=0)

    steps = np.arange(max_len)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(steps, mean_train_loss, label='Train Loss', color='blue', linewidth=2)
    ax1.fill_between(steps, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss,
                     alpha=0.3, color='blue')
    ax1.plot(steps, mean_val_loss, label='Val Loss', color='red', linewidth=2)
    ax1.fill_between(steps, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss,
                     alpha=0.3, color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Average Loss Across Folds (with Std Dev)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(steps, mean_train_acc, label='Train Accuracy', color='green', linewidth=2)
    ax2.fill_between(steps, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                     alpha=0.3, color='green')
    ax2.plot(steps, mean_val_acc, label='Val Accuracy', color='orange', linewidth=2)
    ax2.fill_between(steps, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                     alpha=0.3, color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Average Accuracy Across Folds (with Std Dev)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/average_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/average_metrics.png")
    plt.close()


def plot_confusion_matrix(labels, preds, class_names, save_dir="plots", filename="confusion_matrix.png"):

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Overall K-Fold Results', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/{filename}")
    plt.close()

def plot_per_class_metrics(labels, preds, class_names, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)

    report = classification_report(labels, preds, target_names=class_names, output_dict=True)

    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightgreen')
    ax.bar(x + width, f1, width, label='F1-Score', color='salmon')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/per_class_metrics.png")
    plt.close()


def save_classification_report(labels, preds, class_names, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)

    # Save as text
    report_text = classification_report(labels, preds, target_names=class_names)
    with open(f"{save_dir}/classification_report.txt", 'w') as f:
        f.write("Classification Report - K-Fold Cross Validation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)
    print(f"✓ Saved: {save_dir}/classification_report.txt")

    # Save as table image
    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)

    # Create dataframe
    df_report = pd.DataFrame(report_dict).transpose()
    df_report = df_report.round(3)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(df_report) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_report.values,
                     colLabels=df_report.columns,
                     rowLabels=df_report.index,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * len(df_report.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df_report.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(len(df_report)):
        table[(i + 1, -1)].set_facecolor('#E8E8E8')
        table[(i + 1, -1)].set_text_props(weight='bold')

    plt.title('Classification Report', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{save_dir}/classification_report.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/classification_report.png")
    plt.close()


def generate_all_plots(fold_results, class_names, log_dir="logs", save_dir="plots"):

    print("\n" + "=" * 60)
    print("Generating Plots and Reports")
    print("=" * 60 + "\n")

    # Aggregate all predictions
    all_labels = []
    all_preds = []
    for labels, preds in fold_results:
        all_labels.extend(labels)
        all_preds.extend(preds)

    # Generate plots

    print("1. Plotting average metrics...")
    plot_average_metrics(log_dir, save_dir)

    print("\n2. Generating confusion matrices...")
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)

    print("\n3. Plotting per-class metrics...")
    plot_per_class_metrics(all_labels, all_preds, class_names, save_dir)

    print("\n4. Saving classification report...")
    save_classification_report(all_labels, all_preds, class_names, save_dir)

    print("\n" + "=" * 60)
    print(f"✓ All plots and reports saved to '{save_dir}/' directory")
    print("=" * 60 + "\n")


if __name__ == "__main__":

    print("This module should be imported and called from train_lightning.py")
    print("Add this line at the end of run_kfold() function:")
    print("    generate_all_plots(fold_results, dataset.classes)")