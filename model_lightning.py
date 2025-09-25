import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.networks.nets import DenseNet121
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)


class MedisLightningModel(pl.LightningModule):
    def __init__(self, n_classes=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Model DenseNet121 from MONAI
        self.model = DenseNet121(
            spatial_dims=2,
            in_channels=3,
            out_channels=n_classes
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=n_classes)
        self.precision = MulticlassPrecision(num_classes=n_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=n_classes, average="macro")
        self.f1 = MulticlassF1Score(num_classes=n_classes, average="macro")
        self.confmat = MulticlassConfusionMatrix(num_classes=n_classes)

        # Buffer for confusion matrix
        self.val_preds = []
        self.val_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Validation metrics
        acc = self.val_acc(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # I save values for the Confusion Matrix
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("precision", prec, prog_bar=True)
        self.log("recall", rec, prog_bar=True)
        self.log("f1", f1, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        "Printing Confusion Matrix at the end of each epoch"
        if len(self.val_labels) > 0:
            preds_tensor = torch.tensor(self.val_preds, device=self.device)
            labels_tensor = torch.tensor(self.val_labels, device=self.device)

            cm = self.confmat(preds_tensor, labels_tensor)
            print("\nConfusion Matrix of the epoch end:")
            print(cm.cpu().numpy())

        # Reset for the next epoch
        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
